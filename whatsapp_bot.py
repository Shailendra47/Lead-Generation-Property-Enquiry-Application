"""Twilio WhatsApp questionnaire for Contact Us submissions."""
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from fastapi import BackgroundTasks
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

logger = logging.getLogger("uvicorn.error")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
_whatsapp_sender_setting = re.sub(
    r"^whatsapp:", "", os.getenv("TWILIO_WHATSAPP_FROM", "").strip(), flags=re.I
)
_whatsapp_sender_digits = re.sub(r"\D", "", _whatsapp_sender_setting)
TWILIO_WHATSAPP_FROM = (
    f"+{_whatsapp_sender_digits}"
    if _whatsapp_sender_setting.startswith("+")
    and 8 <= len(_whatsapp_sender_digits) <= 15
    and not _whatsapp_sender_digits.startswith("0")
    else ""
)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
START_TEMPLATE_SID = os.getenv("TWILIO_WHATSAPP_START_TEMPLATE_SID", "").strip()
BOOKING_TEMPLATE_SID = os.getenv("TWILIO_WHATSAPP_BOOKING_TEMPLATE_SID", "").strip()
CUSTOMER_SERVICE_WINDOW = timedelta(hours=24)

# Approved WhatsApp appointment template. Placeholders: name, event, start, end,
# timezone, location/join URL, extra Calendly details.
BOOKING_CONFIRMATION_TEMPLATE_BODY = (
    "Profilent: Hi {{1}}, your {{3}} is confirmed for {{4}} to {{5}} ({{6}}). "
    "Booking email: {{2}}. Location or joining link: {{7}}. {{8}} Reply STOP to opt out."
)

def start_template_variables(full_name: str, email: str, phone: str) -> dict[str, str]:
    """Fill the approved start template with the customer's submitted details."""
    return {"1": full_name, "2": email, "3": phone}


def start_message_text(full_name: str, email: str = "", phone: str = "", requirement: str = "") -> str:
    """Build the in-session greeting and first location question."""
    first_name = full_name.strip().split()[0] if full_name.strip() else "there"
    message = f"Hi {first_name}! 👋 Welcome to *Profilent*.\n\n"
    if email:
        message += f"We have registered your enquiry for {email}.\n\n"
    if requirement.strip():
        message += f"We received your enquiry: \"{requirement.strip()[:300]}\".\n\n"
    message += "To help our property specialist prepare the right options for you, I just need a few quick details.\n\n"
    return message + TEXT_PROMPTS["q5_location"]

def build_prefilled_calendly_url(base_url: str | None, answers: dict) -> str:
    """Append invitee details (name, email, phone) to Calendly URL so fields are pre-filled."""
    url = (base_url or os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min")).strip()
    full_name = (
        answers.get("full_name")
        or f"{answers.get('first_name', '')} {answers.get('last_name', '')}".strip()
        or answers.get("first_name")
        or answers.get("Name")
        or answers.get("customer_name")
        or ""
    ).strip()
    first_name = (answers.get("first_name") or (full_name.split()[0] if full_name else "")).strip()
    last_name = (answers.get("last_name") or (" ".join(full_name.split()[1:]) if len(full_name.split()) > 1 else "")).strip()

    email = (
        answers.get("email")
        or answers.get("Email")
        or answers.get("customer_email")
        or ""
    ).strip()

    phone = (
        answers.get("mobile_number")
        or answers.get("contact_number")
        or answers.get("phone")
        or answers.get("Phone")
        or answers.get("PhoneNumber")
        or answers.get("customer_phone")
        or ""
    ).strip()

    params = {}
    if full_name and full_name.lower() not in {"there", "customer", "valued client"}:
        params["name"] = full_name
    if first_name and first_name.lower() not in {"there", "customer"}:
        params["first_name"] = first_name
    if last_name:
        params["last_name"] = last_name
    if email:
        params["email"] = email
    if phone:
        params["phone"] = phone
        params["a1"] = phone

    if not params:
        return url

    try:
        parts = urlsplit(url)
        query_dict = dict(parse_qsl(parts.query))
        query_dict.update(params)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query_dict), parts.fragment))
    except Exception:
        return url


def format_completion_summary_message(
    answers: dict,
    first_name: str | None = None,
    schedule_url: str | None = None,
    spoke_to_person: bool = False,
    lang: str = "en",
) -> str:
    """Standard unified WhatsApp conversation completion message with full details and both Calendly options."""
    name = (
        first_name
        or answers.get("first_name")
        or (answers.get("full_name", "").split()[0] if answers.get("full_name") else "")
        or ""
    ).strip()
    if name.lower() in {"there", "customer"}:
        name = ""

    full_name_val = (
        answers.get("full_name")
        or f"{answers.get('first_name', '')} {answers.get('last_name', '')}".strip()
        or name
        or "Valued Client"
    ).strip()
    phone_val = (
        answers.get("mobile_number")
        or answers.get("contact_number")
        or answers.get("phone")
        or answers.get("Phone")
        or answers.get("PhoneNumber")
        or ""
    ).strip()
    email_val = (
        answers.get("email")
        or answers.get("Email")
        or answers.get("customer_email")
        or ""
    ).strip()

    raw_meeting_url = (
        schedule_url
        or os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min")
    ).strip()
    raw_visit_url = os.getenv("CALENDLY_VISIT_URL", "https://calendly.com/rudrasinghdeo369/30min").strip()

    ctx = {
        **answers,
        "full_name": full_name_val,
        "first_name": name or answers.get("first_name", ""),
        "email": email_val,
        "phone": phone_val,
        "mobile_number": phone_val,
    }

    meeting_url = build_prefilled_calendly_url(raw_meeting_url, ctx)
    visit_url = build_prefilled_calendly_url(raw_visit_url, ctx)

    loc = answers.get("location_preference") or "Not specified"
    timeline = answers.get("planning_to_buy") or "Not specified"
    budget = answers.get("budget_range") or "Not specified"
    payment = answers.get("payment_option") or "Not specified"
    bhk = answers.get("bhk_preference") or "Not specified"

    req = (
        answers.get("remarks_requirements")
        or answers.get("requirement")
        or answers.get("special_requirements")
        or answers.get("form_requirement")
        or ("None" if lang != "hi" else "कोई नहीं")
    ).strip()
    if not req or req.lower() in {"none", "skip", "n/a", "-", "null", "no"}:
        req = "None" if lang != "hi" else "कोई नहीं"

    if lang == "hi":
        greet = f"✅ धन्यवाद, {name}! आपकी सभी आवश्यकताएं और बातचीत का विवरण नोट कर लिया गया है:" if name else "✅ धन्यवाद! आपकी सभी आवश्यकताएं और बातचीत का विवरण नोट कर लिया गया है:"
        bullets = [
            f"• ग्राहक का नाम: {full_name_val}",
        ]
        if email_val:
            bullets.append(f"• ईमेल (Email): {email_val}")
        if phone_val:
            bullets.append(f"• फ़ोन नंबर: {phone_val}")
        bullets.extend([
            f"• पसंदीदा क्षेत्र (Location): {loc}",
            f"• प्रॉपर्टी कॉन्फ़िगरेशन (BHK): {bhk}",
            f"• बजट रेंज (Budget): {budget}",
            f"• ख़रीदने की समयसीमा (Timeline): {timeline}",
            f"• भुगतान का तरीका (Payment): {payment}",
            f"• विशेष आवश्यकताएं (Requirements): {req}",
        ])
        spoke_prefix = "हमारे सीनियर प्रॉपर्टी स्पेशलिस्ट जल्द ही आपसे संपर्क करेंगे।\n\n" if spoke_to_person else ""
        note = "हमारी प्रॉपर्टी स्पेशलिस्ट टीम आपकी सभी प्राथमिकताओं की समीक्षा करेगी और आपके लिए सर्वोत्तम विकल्प तैयार करेगी।"
        cta = (
            "🗓️ *आगे बढ़ने के लिए कृपया नीचे दिए गए विकल्पों में से चुनें:*\n\n"
            "1️⃣ *परामर्श मीटिंग शेड्यूल करें (Schedule a Meeting):*\n"
            f"👉 {meeting_url}\n\n"
            "2️⃣ *प्रॉपर्टी साइट विज़िट शेड्यूल करें (Schedule a Visit):*\n"
            f"👉 {visit_url}"
        )
    elif lang == "hinglish":
        greet = f"✅ Dhanyawaad, {name}! Aapki conversation details aur requirements note kar li gayi hain:" if name else "✅ Dhanyawaad! Aapki conversation details aur requirements note kar li gayi hain:"
        bullets = [
            f"• Customer Name: {full_name_val}",
        ]
        if email_val:
            bullets.append(f"• Email: {email_val}")
        if phone_val:
            bullets.append(f"• Phone Number: {phone_val}")
        bullets.extend([
            f"• Preferred Location: {loc}",
            f"• Property Type (BHK): {bhk}",
            f"• Budget Range: {budget}",
            f"• Buying Timeline: {timeline}",
            f"• Payment Option: {payment}",
            f"• Special Requirements: {req}",
        ])
        spoke_prefix = "Hamare senior property specialist aapse jaldi hi contact karenge.\n\n" if spoke_to_person else ""
        note = "Hamari property specialist team aapki requirements review karke best options ready karegi."
        cta = (
            "🗓️ *Aage proceed karne ke liye please neeche diye gaye options me se choose karein:*\n\n"
            "1️⃣ *Schedule a Meeting:*\n"
            f"👉 {meeting_url}\n\n"
            "2️⃣ *Schedule a Visit:*\n"
            f"👉 {visit_url}"
        )
    else:
        greet = f"✅ Thank you, {name}! Here is the complete summary of your property enquiry:" if name else "✅ Thank you! Here is the complete summary of your property enquiry:"
        bullets = [
            f"• Customer Name: {full_name_val}",
        ]
        if email_val:
            bullets.append(f"• Email: {email_val}")
        if phone_val:
            bullets.append(f"• Phone Number: {phone_val}")
        bullets.extend([
            f"• Location Preference: {loc}",
            f"• Configuration / BHK: {bhk}",
            f"• Budget Range: {budget}",
            f"• Buying Timeline: {timeline}",
            f"• Payment Option: {payment}",
            f"• Special Requirements: {req}",
        ])
        spoke_prefix = "A senior property specialist will contact you shortly.\n\n" if spoke_to_person else ""
        note = "Our specialist team will review your requirements and prepare hand-picked options."
        cta = (
            "🗓️ *Please choose one of the options below to proceed:*\n\n"
            "1️⃣ *Schedule a Meeting:*\n"
            f"👉 {meeting_url}\n\n"
            "2️⃣ *Schedule a Visit:*\n"
            f"👉 {visit_url}"
        )

    bullet_block = "\n".join(bullets)
    return f"{spoke_prefix}{greet}\n\n{bullet_block}\n\n{note}\n\n{cta}"


def whatsapp_configuration() -> dict:
    public_origin = PUBLIC_BASE_URL.casefold()
    public_https_ready = (
        public_origin.startswith("https://")
        and "localhost" not in public_origin
        and "127.0.0.1" not in public_origin
    )
    settings = (
        ("TWILIO_ACCOUNT_SID", bool(re.fullmatch(r"AC[0-9a-fA-F]{32}", TWILIO_ACCOUNT_SID or ""))),
        ("TWILIO_AUTH_TOKEN", bool(TWILIO_AUTH_TOKEN and "replace_with" not in TWILIO_AUTH_TOKEN.casefold() and "<" not in TWILIO_AUTH_TOKEN)),
        ("TWILIO_WHATSAPP_FROM", bool(re.fullmatch(r"\+[1-9][0-9]{7,14}", TWILIO_WHATSAPP_FROM or ""))),
        ("TWILIO_WHATSAPP_START_TEMPLATE_SID", bool(re.fullmatch(r"HX[0-9a-fA-F]{32}", START_TEMPLATE_SID or ""))),
    )
    missing = [name for name, configured in settings if not configured]
    if re.sub(r"\D", "", TWILIO_WHATSAPP_FROM) == "14155238886":
        missing.append("registered WhatsApp sender (custom templates are unavailable in Sandbox)")
    return {
        "outbound_configured": not missing,
        "inbound_webhook_configured": bool(TWILIO_AUTH_TOKEN and public_https_ready),
        "missing_outbound_settings": missing,
        "inbound_webhook_url": f"{PUBLIC_BASE_URL}/webhooks/whatsapp" if PUBLIC_BASE_URL else "",
    }

LISTS = {
    "q6_planning": {
        "sid": os.getenv("TWILIO_LIST_PLANNING_SID", ""),
        "question": "When are you planning to buy?",
        "rows": [
            ("planning_immediately", "Immediately", "Immediately"),
            ("planning_1_month", "Within 1 Month", "Within 1 Month"),
            ("planning_3_months", "Within 3 Months", "Within 3 Months"),
            ("planning_6_months", "Within 6 Months", "Within 6 Months"),
            ("planning_12_months", "Within 12 Months", "Within 12 Months"),
            ("planning_over_1_year", "More Than 1 Year", "More Than 1 Year"),
            ("planning_exploring", "Just Exploring", "Just Exploring / Researching"),
        ],
    },
    "q7_budget": {
        "sid": os.getenv("TWILIO_LIST_BUDGET_SID", ""),
        "question": "What's your budget range?",
        "rows": [
            ("budget_25_lakh", "Up to 25 Lakh", "Up to 25 lakh"),
            ("budget_25_50_lakh", "25 - 50 Lakh", "25 lakh - 50 lakh"),
            ("budget_50_lakh_1_crore", "50 Lakh - 1 Crore", "50 lakh - 1 crore"),
            ("budget_1_2_crore", "1 - 2 Crore", "1 crore - 2 crore"),
            ("budget_over_2_crore", "Above 2 Crore", "Above 2 crore"),
        ],
    },
    "q8_category": {
        "sid": os.getenv("TWILIO_LIST_PAYMENT_CATEGORY_SID", ""),
        "question": "How would you like to pay? Choose a category:",
        "rows": [
            ("pay_loan", "Loan / Bank Finance", "Loan / Bank Finance"),
            ("pay_cash", "Cash / Down Payment", "Cash / Down Payment"),
            ("pay_developer", "Developer Plans", "Developer Plans"),
            ("pay_nri", "NRI Payment Plan", "NRI Payment Plan (for Overseas Buyers)"),
        ],
    },
    "q8_loan": {
        "sid": os.getenv("TWILIO_LIST_PAYMENT_LOAN_SID", ""),
        "question": "Choose your loan or bank-finance option:",
        "rows": [
            ("loan_bank_emi", "Bank Finance (EMI)", "Bank Finance with EMI"),
            ("loan_mortgage", "Home Loan (Mortgage)", "Home Loan (Mortgage)"),
            ("loan_part_cash", "Part Cash + Home Loan", "Part Cash + Home Loan"),
            ("loan_joint", "Joint Home Loan", "Joint Home Loan (Co-applicant Financing)"),
        ],
    },
    "q8_cash": {
        "sid": os.getenv("TWILIO_LIST_PAYMENT_CASH_SID", ""),
        "question": "Choose your cash or down-payment option:",
        "rows": [
            ("cash_full", "Full Cash Payment", "Full Cash Payment"),
            ("cash_down_payment", "Down Payment Plan", "Down Payment Plan"),
            ("cash_balance_registration", "Balance at Registration", "Part Payment with Balance at Registration"),
        ],
    },
    "q8_developer": {
        "sid": os.getenv("TWILIO_LIST_PAYMENT_DEVELOPER_SID", ""),
        "question": "Choose your developer payment plan:",
        "rows": [
            ("dev_clp", "Construction-Linked", "Construction-Linked Payment Plan (CLP)"),
            ("dev_plp", "Possession-Linked", "Possession-Linked Payment Plan (PLP)"),
            ("dev_installment", "Installment Plan", "Installment Payment Plan"),
            ("dev_financing", "Developer Financing", "Developer Financing"),
            ("dev_rent_to_own", "Rent-to-Own", "Rent-to-Own Scheme"),
            ("dev_subvention", "Subvention (Pre-EMI)", "Subvention Scheme (Developer Pays Pre-EMI)"),
            ("dev_flexible", "Flexible Payment Plan", "Flexible Payment Plan"),
        ],
    },
    "q8_payment": {
        "sid": os.getenv("TWILIO_LIST_PAYMENT_SID", ""),
        "question": "How would you like to pay? Choose an option:",
        "rows": [
            ("pay_bank_emi", "Bank Finance (EMI)", "Bank Finance with EMI"),
            ("pay_cash_full", "Full Cash Payment", "Full Cash Payment"),
            ("pay_home_loan", "Home Loan (Mortgage)", "Home Loan (Mortgage)"),
            ("pay_down_payment", "Down Payment Plan", "Down Payment Plan"),
            ("pay_clp", "Construction-Linked (CLP)", "Construction-Linked Payment Plan (CLP)"),
            ("pay_plp", "Possession-Linked (PLP)", "Possession-Linked Payment Plan (PLP)"),
            ("pay_installment", "Installment Plan", "Installment Payment Plan"),
            ("pay_developer_fin", "Developer Financing", "Developer Financing"),
            ("pay_rent_to_own", "Rent-to-Own Scheme", "Rent-to-Own Scheme"),
            ("pay_part_cash_loan", "Part Cash + Home Loan", "Part Cash + Home Loan"),
            ("pay_balance_reg", "Balance at Registration", "Part Payment with Balance at Registration"),
            ("pay_flexible", "Flexible Payment Plan", "Flexible Payment Plan"),
            ("pay_subvention", "Subvention (Pre-EMI)", "Subvention Scheme (Developer Pays Pre-EMI)"),
            ("pay_joint_loan", "Joint Home Loan", "Joint Home Loan (Co-applicant Financing)"),
            ("pay_nri_plan", "NRI Payment Plan", "NRI Payment Plan (for Overseas Buyers)"),
        ],
    },
    "q9_bhk": {
        "sid": os.getenv("TWILIO_LIST_BHK_SID", ""),
        "question": "What type of home are you looking for?",
        "rows": [
            ("bhk_1_rk", "1 RK", "1 RK"),
            ("bhk_1", "1 BHK", "1 BHK"),
            ("bhk_2", "2 BHK", "2 BHK"),
            ("bhk_2_5", "2.5 BHK", "2.5 BHK"),
            ("bhk_3", "3 BHK", "3 BHK"),
            ("bhk_3_5", "3.5 BHK", "3.5 BHK"),
            ("bhk_4", "4 BHK", "4 BHK"),
            ("bhk_5_plus", "5 BHK+", "5 BHK+"),
            ("bhk_villa", "Villa", "Villa"),
            ("bhk_penthouse", "Penthouse", "Penthouse"),
        ],
    },
}

QUICK_REPLY_MOBILE_SID = os.getenv("TWILIO_QUICK_REPLY_MOBILE_SID", "")
_LIST_LOOKUPS = {
    stage: {
        "by_id": {row_id: (label, value) for row_id, label, value in menu["rows"]},
        "by_label": {label.casefold(): (label, value) for _, label, value in menu["rows"]},
    }
    for stage, menu in LISTS.items()
}

TEXT_PROMPTS = {
    "q1_first_name": "Hi! 👋 Let's find your perfect property. What's your *first name*?",
    "q2_last_name": "Thanks, {first_name}! And your *last name*? (Type *skip* to continue)",
    "q3_email": "Please share your *email address*.",
    "q3_email_invalid": "That doesn't look right. Please enter a valid email, e.g. name@example.com",
    "q4_mobile_other": "Please type another 10-digit number.",
    "q5_location": (
        "Which area are you interested in?\n"
        "1. Haldwani (Rampur Rd, Kaladhungi Rd, Bithoria, Mukhani)\n"
        "2. Nainital / Bhimtal\n"
        "3. Almora\n"
        "4. Other (please type the area)"
    ),
    "q6_planning": (
        "When are you planning to buy?\n"
        "1. Immediately\n"
        "2. Within 1 Month\n"
        "3. Within 3 Months\n"
        "4. Within 6 Months\n"
        "5. Within 12 Months\n"
        "6. More Than 1 Year\n"
        "7. Just Exploring / Researching"
    ),
    "q7_budget": (
        "What is your budget range?\n"
        "1. Up to 25 lakh\n"
        "2. 25 lakh - 50 lakh\n"
        "3. 50 lakh - 1 crore\n"
        "4. 1 crore - 2 crore\n"
        "5. Above 2 crore"
    ),
    "q8_payment": (
        "How would you like to pay? Please choose your payment option:\n"
        "1. Bank Finance with EMI\n"
        "2. Full Cash Payment\n"
        "3. Home Loan (Mortgage)\n"
        "4. Down Payment Plan\n"
        "5. Construction-Linked Payment Plan (CLP)\n"
        "6. Possession-Linked Payment Plan (PLP)\n"
        "7. Installment Payment Plan\n"
        "8. Developer Financing\n"
        "9. Rent-to-Own Scheme\n"
        "10. Part Cash + Home Loan\n"
        "11. Part Payment with Balance at Registration\n"
        "12. Flexible Payment Plan\n"
        "13. Subvention Scheme (Developer Pays Pre-EMI)\n"
        "14. Joint Home Loan (Co-applicant Financing)\n"
        "15. NRI Payment Plan (for Overseas Buyers)"
    ),
    "q9_bhk": (
        "What type of home / configuration are you looking for?\n"
        "1. 1 RK\n"
        "2. 1 BHK\n"
        "3. 2 BHK\n"
        "4. 2.5 BHK\n"
        "5. 3 BHK\n"
        "6. 3.5 BHK\n"
        "7. 4 BHK\n"
        "8. 5 BHK+\n"
        "9. Villa\n"
        "10. Penthouse"
    ),
    "q10_remarks": "Any *remarks or requirements* for your property search? (Type *skip* if none)",
}

STAGE_PROMPT = {
    "q1_first_name": "q1_first_name",
    "q2_last_name": "q2_last_name",
    "q3_email": "q3_email",
    "q4_mobile": "q4_mobile",
    "q4_mobile_other": "q4_mobile_other",
    "q5_location": "q5_location",
    "q6_planning": "q6_planning",
    "q7_budget": "q7_budget",
    "q8_category": "q8_category",
    "q8_payment": "q8_payment",
    "q8_loan": "q8_loan",
    "q8_cash": "q8_cash",
    "q8_developer": "q8_developer",
    "q9_bhk": "q9_bhk",
    "q10_remarks": "q10_remarks",
}


def twiml(message: str = "") -> Response:
    response = MessagingResponse()
    if message:
        response.message(message)
    xml_bytes = str(response).encode("utf-8")
    return Response(
        content=xml_bytes,
        media_type="text/xml",
        headers={"Content-Type": "text/xml; charset=utf-8"},
    )


def _phone_suffix(phone: str) -> str:
    return re.sub(r"\D", "", phone)[-4:]


_twilio_client_lock = threading.Lock()
_cached_twilio_client = None
_cached_account_sid = ""
_cached_auth_token = ""
_last_cred_check = 0.0

def _get_twilio_client() -> tuple[Client | None, str]:
    """Return a cached, persistent Twilio Client with low-overhead credential check."""
    global _cached_twilio_client, _cached_account_sid, _cached_auth_token, _last_cred_check
    now = time.monotonic()
    if now - _last_cred_check > 5.0 or not _cached_auth_token:
        load_dotenv(override=False)
        _last_cred_check = now
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip() or TWILIO_ACCOUNT_SID
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip() or TWILIO_AUTH_TOKEN
    from_raw = os.getenv("TWILIO_WHATSAPP_FROM", "").strip() or TWILIO_WHATSAPP_FROM
    from_sender = re.sub(r"^whatsapp:", "", from_raw, flags=re.I)

    if not (account_sid and auth_token and from_sender):
        return None, ""

    with _twilio_client_lock:
        if _cached_twilio_client is None or _cached_account_sid != account_sid or _cached_auth_token != auth_token:
            _cached_twilio_client = Client(account_sid, auth_token)
            _cached_account_sid = account_sid
            _cached_auth_token = auth_token
        return _cached_twilio_client, from_sender


def send_content(
    recipient_phone: str,
    content_sid: str,
    variables: dict | None = None,
    *,
    purpose: str = "template",
) -> str | None:
    """Send from the configured business sender to the customer's WhatsApp number."""
    content_sid = str(content_sid or "").strip()
    if not re.fullmatch(r"HX[0-9a-fA-F]{32}", content_sid):
        logger.error("WhatsApp template send skipped: Content SID is missing or malformed.")
        return None
    client, from_sender = _get_twilio_client()
    if not client or not content_sid:
        logger.error("WhatsApp message skipped: Twilio credentials, sender, or Content SID is missing.")
        return None
    try:
        kwargs = {
            "from_": f"whatsapp:{from_sender}",
            "to": f"whatsapp:{recipient_phone}",
            "content_sid": content_sid,
        }
        if variables:
            kwargs["content_variables"] = json.dumps(variables)
        message = client.messages.create(**kwargs)
        return message.sid
    except Exception as exc:
        err_code = getattr(exc, "code", None)
        if err_code in (572002, 21608):
            logger.info(
                "WhatsApp template outbound push skipped for %s (Twilio Trial restriction, err_code=%s). Waiting for customer inbound.",
                _phone_suffix(recipient_phone),
                err_code,
            )
        else:
            logger.warning(
                "WhatsApp template not sent (purpose=%s, content_sid=%s, twilio_error_code=%s, phone_suffix=%s): %s",
                purpose,
                content_sid,
                err_code,
                _phone_suffix(recipient_phone),
                str(exc).splitlines()[0] if str(exc) else "Template error",
            )
        return None


def send_text(to_phone: str, body: str) -> str | None:
    """Send an in-session WhatsApp message, used for the Twilio Sandbox."""
    client, from_sender = _get_twilio_client()
    if not client or not body:
        logger.error("WhatsApp text skipped: Twilio credentials, sender, or message is missing.")
        return None
    try:
        clean_to = re.sub(r"[^\d+]", "", to_phone)
        if not clean_to.startswith("+"):
            clean_to = f"+{clean_to}"
        clean_from = re.sub(r"[^\d+]", "", from_sender)
        if not clean_from.startswith("+"):
            clean_from = f"+{clean_from}"
        message = client.messages.create(
            from_=f"whatsapp:{clean_from}",
            to=f"whatsapp:{clean_to}",
            body=body,
        )
        logger.info("WhatsApp text sent via Twilio to %s, sid=%s", clean_to, message.sid)
        return message.sid
    except Exception as exc:
        err_msg = str(exc)
        if "ContentSid" in err_msg or getattr(exc, "code", None) in (21654, 92005, 63016, 63018):
            logger.warning(
                "WhatsApp text not sent (phone suffix=%s): 24h session is closed; customer must message first via prefill link (%s)",
                _phone_suffix(to_phone),
                err_msg.splitlines()[0] if err_msg else "ContentSid Required",
            )
        else:
            logger.warning(
                "Could not send WhatsApp text (phone suffix=%s): %s",
                _phone_suffix(to_phone),
                err_msg.splitlines()[0] if err_msg else str(exc),
            )
        return None


def send_content_or_text(
    recipient_phone: str,
    content_sid: str,
    fallback_text: str,
    variables: dict | None = None,
    *,
    purpose: str = "interactive_questionnaire",
) -> None:
    """Try an approved interactive message, then use an in-session text fallback."""
    if send_content(recipient_phone, content_sid, variables, purpose=purpose):
        return
    if fallback_text:
        send_text(recipient_phone, fallback_text)


def booking_confirmation_text(variables: dict) -> str:
    """Fill the appointment prompt with Calendly date, time, and timezone."""
    text = BOOKING_CONFIRMATION_TEMPLATE_BODY
    for key in ("1", "2", "3", "4", "5", "6", "7", "8"):
        text = text.replace("{{" + key + "}}", str(variables.get(key, "")).strip())
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def booking_window_open(conversation: dict | None) -> bool:
    """True when the customer's last inbound WhatsApp message is still inside 24 hours."""
    if not conversation:
        return False
    # Meta WhatsApp 24-hour window requires an inbound customer message
    timestamp = conversation.get("last_inbound_at")
    if not timestamp:
        return False
    if getattr(timestamp, "tzinfo", None) is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timestamp <= CUSTOMER_SERVICE_WINDOW


def send_booking_confirmation(recipient_phone: str, variables: dict, window_open: bool) -> str | None:
    """Send full booking details using an approved template or open-session text."""
    sender_digits = re.sub(r"\D", "", TWILIO_WHATSAPP_FROM)
    if sender_digits == "14155238886":
        # The Sandbox has only a small set of account-provided templates. A
        # hard-coded example Content SID is not guaranteed to belong to this
        # account, so send full booking details as free-form text only while
        # the customer's 24-hour service window is open.
        if window_open:
            return send_text(recipient_phone, booking_confirmation_text(variables))
        logger.error(
            "Sandbox booking confirmation skipped: the customer service window is closed; "
            "use a registered WhatsApp sender and an approved booking template."
        )
        return None
    if BOOKING_TEMPLATE_SID:
        message_sid = send_content(
            recipient_phone,
            BOOKING_TEMPLATE_SID,
            variables,
            purpose="booking_confirmation",
        )
        if message_sid or not window_open:
            return message_sid
        logger.warning("Calendly template send failed in an open customer-service window; trying text.")
    if window_open:
        return send_text(recipient_phone, booking_confirmation_text(variables))
    logger.error("Calendly confirmation skipped: approved booking template is not configured.")
    return None


async def start_questionnaire(
    conversations,
    phone: str,
    contact_id: str,
    background_tasks: BackgroundTasks,
    *,
    full_name: str = "",
    email: str = "",
    requirement: str = "",
):
    prior_conversation = await conversations.find_one({"phone": phone})
    sandbox_window_open = booking_window_open(prior_conversation)
    name_parts = full_name.strip().split(maxsplit=1)
    answers = {
        "full_name": full_name,
        "first_name": name_parts[0] if name_parts else "",
        "last_name": name_parts[1] if len(name_parts) > 1 else "",
        "email": email,
        "mobile_number": phone,
    }
    now_utc = datetime.now(timezone.utc)

    # 1. Match by specific contact submission if provided
    target_conv = None
    if contact_id:
        target_conv = await conversations.find_one({"contact_id": contact_id})

    # 2. If not matched by contact_id, check for an active in-progress conversation for this phone
    if not target_conv:
        active_c = await conversations.find_one(
            {"phone": phone, "status": {"$in": ["in_progress", "awaiting_location", "awaiting_customer_start"]}},
            sort=[("updated_at", -1)],
        )
        if active_c:
            a_ans = active_c.get("answers") or {}
            a_email = (active_c.get("email") or a_ans.get("email") or "").strip().lower()
            a_name = (active_c.get("full_name") or a_ans.get("full_name") or "").strip().lower()
            c_email = email.strip().lower()
            c_name = full_name.strip().lower()
            # If same customer, continue on same conversation
            if (c_email and a_email and c_email == a_email) or (c_name and a_name and c_name == a_name):
                target_conv = active_c

    # 3. If target conversation exists, update it by _id.
    # Otherwise, insert a BRAND NEW conversation document so each user gets their own separate row!
    if target_conv:
        conv_id = target_conv["_id"]
        await conversations.update_one(
            {"_id": conv_id},
            {
                "$set": {
                    "phone": phone,
                    "contact_id": contact_id,
                    "first_name": answers["first_name"],
                    "last_name": answers["last_name"],
                    "full_name": full_name,
                    "email": email,
                    "stage": "q_ai_flow",
                    "answers": answers,
                    "form_requirement": requirement.strip(),
                    "requirement_messages": [requirement.strip()] if requirement.strip() else [],
                    "status": "awaiting_location",
                    "updated_at": now_utc,
                },
                "$unset": {
                    "completed_at": "",
                    "property_lead_id": "",
                    "schedule_url": "",
                },
            },
        )
    else:
        new_doc = {
            "phone": phone,
            "contact_id": contact_id,
            "first_name": answers["first_name"],
            "last_name": answers["last_name"],
            "full_name": full_name,
            "email": email,
            "stage": "q_ai_flow",
            "answers": answers,
            "form_requirement": requirement.strip(),
            "requirement_messages": [requirement.strip()] if requirement.strip() else [],
            "status": "awaiting_location",
            "created_at": now_utc,
            "updated_at": now_utc,
            "messages": [],
        }
        res = await conversations.insert_one(new_doc)
        conv_id = res.inserted_id

    sender_digits = re.sub(r"\D", "", TWILIO_WHATSAPP_FROM)
    is_sandbox = sender_digits == "14155238886"

    message_sid = None
    sent_body = ""
    # In Sandbox mode (+14155238886), outbound server pushes are disallowed by Twilio.
    # The customer opens the pre-filled WhatsApp link and sends 'Hi' to start the session.
    if not is_sandbox and START_TEMPLATE_SID:
        try:
            message_sid = await run_in_threadpool(
                send_content,
                phone,
                START_TEMPLATE_SID,
                start_template_variables(full_name, email, phone),
                purpose="questionnaire_start",
            )
            sent_body = "Sent WhatsApp Template Greeting"
        except Exception as e:
            logger.warning("WhatsApp send_content error for %s: %s", phone, e)

    if not is_sandbox and not message_sid:
        try:
            fallback_body = start_message_text(full_name, email=email, requirement=requirement)
            message_sid = await run_in_threadpool(send_text, phone, fallback_body)
            sent_body = fallback_body
        except Exception as e:
            logger.warning("WhatsApp send_text error for %s: %s", phone, e)

    if message_sid:
        now_utc = datetime.now(timezone.utc)
        await conversations.update_one(
            {"_id": conv_id},
            {
                "$set": {
                    "status": "awaiting_location",
                    "start_message_sid": message_sid,
                    "updated_at": now_utc,
                },
                "$push": {
                    "messages": {
                        "role": "assistant",
                        "content": sent_body,
                        "timestamp": now_utc.isoformat(),
                    }
                },
            },
        )
        return {"started": True, "status": "accepted", "message_sid": message_sid}
    else:
        await conversations.update_one(
            {"_id": conv_id},
            {"$set": {
                "status": "awaiting_customer_start",
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        return {"started": True, "sandbox": is_sandbox, "status": "awaiting_customer_start"}


def menu_fallback(stage: str) -> str:
    menu = LISTS[stage]
    rows = menu["rows"]
    choices = "\n".join(f"{i}. {title}" for i, (_, title, _) in enumerate(rows, 1))
    return f"{menu['question']}\n{choices}\n\nReply with the option number."


def dispatch_stage(stage: str, phone: str, answers: dict, background_tasks: BackgroundTasks) -> str:
    if stage in LISTS:
        content_sid = LISTS[stage]["sid"]
        if content_sid:
            background_tasks.add_task(
                send_content_or_text,
                phone,
                content_sid,
                menu_fallback(stage),
                purpose=stage,
            )
            return ""
        return menu_fallback(stage)
    if stage == "q4_mobile":
        if QUICK_REPLY_MOBILE_SID:
            background_tasks.add_task(
                send_content_or_text,
                phone,
                QUICK_REPLY_MOBILE_SID,
                f"Is *{phone}* the best number to reach you? Reply *Yes*, or type another 10-digit number.",
                {"1": phone},
                purpose="mobile_confirmation",
            )
            return ""
        return f"Is *{phone}* the best number to reach you? Reply *Yes*, or type another 10-digit number."
    prompt_key = STAGE_PROMPT.get(stage)
    if not prompt_key:
        return ""
    if stage == "q2_last_name":
        return TEXT_PROMPTS[prompt_key].format(first_name=answers.get("first_name", "there"))
    return TEXT_PROMPTS[prompt_key]


def list_choice(stage: str, form) -> tuple[str, str] | None:
    menu = LISTS[stage]
    lookups = _LIST_LOOKUPS[stage]
    for candidate in (
        str(form.get("ButtonPayload", "")).strip(),
        str(form.get("Body", "")).strip(),
        str(form.get("ButtonText", "")).strip(),
    ):
        if candidate in lookups["by_id"]:
            return lookups["by_id"][candidate]
        if candidate.casefold() in lookups["by_label"]:
            return lookups["by_label"][candidate.casefold()]
        if candidate.isdigit():

            index = int(candidate) - 1

            if 0 <= index < len(menu["rows"]):

                _, label, value = menu["rows"][index]

                return label, value

    return None





def find_list_stage(stage: str) -> str | None:
    if stage in LISTS:
        return stage
    return None

