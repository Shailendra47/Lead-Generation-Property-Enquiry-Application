"""Multi-Device WhatsApp Service & Handheld Device Dispatcher
============================================================
Handles phone number validation, server-side push to customer handheld phones,
and multi-device conversation forwarding (sales team & companion devices).

Architecture:
1. Customer Handheld Push:
   - When a customer fills the form on ANY device (e.g., PC, laptop, or another phone),
     the browser does NOT need to open WhatsApp locally.
   - Instead, the backend server pushes a WhatsApp message directly over the internet
     (via Twilio API) to the customer's phone number.
   - The message pops up in the downloaded WhatsApp application on the customer's handheld phone!

2. Multi-Device Synchronization:
   - Customer-side: WhatsApp natively syncs the conversation across up to 4 linked
     companion devices (Phone, Tablet, WhatsApp Web, Desktop).
   - Team-side: The server can mirror/forward incoming customer replies to multiple
     sales team devices simultaneously (MULTI_DEVICE_ALERT_PHONES).
"""

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from twilio.rest import Client

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/device", tags=["Multi-Device WhatsApp"])

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "+14155238886")
CALENDLY_URL = os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min")


# -------------------------------------------------------------
# 1. Handheld Device Number Validation & Formatting
# -------------------------------------------------------------
def normalize_device_phone(raw_phone: str, default_country_code: str = "+91") -> str:
    """Normalize any phone string into canonical international E.164 format (+91XXXXXXXXXX).
    Detects Indian 10-digit mobile numbers or international numbers.
    """
    if not raw_phone:
        return ""
    cleaned = re.sub(r"[\s()\-.]", "", str(raw_phone).strip())
    if cleaned.startswith("+"):
        digits = re.sub(r"\D", "", cleaned)
        return f"+{digits}" if len(digits) >= 8 else ""

    digits = re.sub(r"\D", "", cleaned).lstrip("0")
    if len(digits) == 10 and digits[0] in "6789":
        cc = default_country_code if default_country_code.startswith("+") else f"+{default_country_code}"
        return f"{cc}{digits}"
    elif len(digits) == 12 and digits.startswith("91"):
        return f"+{digits}"
    elif 8 <= len(digits) <= 15:
        cc = default_country_code if default_country_code.startswith("+") else f"+{default_country_code}"
        return f"{cc}{digits}"
    return ""


def is_valid_mobile_number(phone_e164: str) -> bool:
    """Check if the normalized number conforms to valid mobile phone standards."""
    if not phone_e164 or not phone_e164.startswith("+"):
        return False
    digits = phone_e164[1:]
    # Indian mobile check (starts with +91 followed by 6, 7, 8, or 9 and 9 more digits)
    if digits.startswith("91"):
        local = digits[2:]
        return len(local) == 10 and local[0] in "6789"
    # General international mobile range: 8 to 15 digits
    return 8 <= len(digits) <= 15


# -------------------------------------------------------------
# 2. Direct Cloud-to-Handheld Device WhatsApp Dispatch
# -------------------------------------------------------------
def get_twilio_client() -> Optional[Client]:
    """Initialize Twilio REST client if valid credentials exist."""
    sid = (TWILIO_ACCOUNT_SID or "").strip()
    token = (TWILIO_AUTH_TOKEN or "").strip()
    if sid.startswith("AC") and len(sid) == 34 and token and token != "your_auth_token":
        try:
            return Client(sid, token)
        except Exception as exc:
            logger.warning("Could not initialize Twilio client: %s", exc)
    return None


def dispatch_to_customer_handheld(
    customer_phone: str,
    message_body: str,
) -> Dict[str, Any]:
    """Send an outbound WhatsApp message directly to the customer's handheld phone via Twilio API.
    
    The customer does not need to have the website open on their phone.
    The message will arrive in the native WhatsApp app downloaded on their mobile device.
    """
    normalized = normalize_device_phone(customer_phone)
    if not is_valid_mobile_number(normalized):
        return {
            "success": False,
            "error": f"Invalid mobile phone number format: {customer_phone}. Must be a valid mobile with country code.",
            "phone": normalized,
        }

    client = get_twilio_client()
    if not client:
        return {
            "success": False,
            "error": "Twilio credentials are not configured or invalid in .env",
            "phone": normalized,
        }

    sender = TWILIO_WHATSAPP_FROM.strip()
    if not sender.startswith("+"):
        sender = f"+{re.sub(r'\\D', '', sender)}"
    from_wa = f"whatsapp:{sender}"
    to_wa = f"whatsapp:{normalized}"

    sender_digits = re.sub(r"\D", "", sender)
    is_sandbox = sender_digits == "14155238886"

    try:
        msg = client.messages.create(
            from_=from_wa,
            to=to_wa,
            body=message_body,
        )
        logger.info(
            "Dispatched WhatsApp to handheld device %s (SID: %s, sandbox=%s)",
            normalized,
            msg.sid,
            is_sandbox,
        )
        return {
            "success": True,
            "message_sid": msg.sid,
            "phone": normalized,
            "status": msg.status,
            "is_sandbox": is_sandbox,
            "device_delivery_note": (
                "Message sent! If using Twilio Sandbox (+14155238886), the target phone must have joined the sandbox once."
                if is_sandbox
                else "Delivered directly to customer's handheld device via WhatsApp Business API."
            ),
        }
    except Exception as exc:
        err_text = str(exc)
        logger.error("Failed to dispatch WhatsApp to %s: %s", normalized, err_text)
        return {
            "success": False,
            "error": err_text,
            "phone": normalized,
            "is_sandbox": is_sandbox,
            "code": getattr(exc, "code", None),
        }


# -------------------------------------------------------------
# 3. Multi-Device Broadcast (Team / Sales Agents Devices)
# -------------------------------------------------------------
def get_team_device_numbers() -> List[str]:
    """Retrieve list of team member / sales agent phone numbers to receive conversation mirrors."""
    raw_list = os.getenv("MULTI_DEVICE_ALERT_PHONES", "")
    if not raw_list:
        # Fall back to AGENT_PHONE if configured
        agent = os.getenv("AGENT_PHONE", "").strip()
        if agent:
            raw_list = agent

    phones = []
    for item in raw_list.split(","):
        norm = normalize_device_phone(item.strip())
        if norm and is_valid_mobile_number(norm) and norm not in phones:
            phones.append(norm)
    return phones


def broadcast_to_multiple_devices(
    alert_text: str,
    target_phones: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Broadcast an alert or customer conversation snippet to multiple handheld devices
    (e.g., Sales Rep 1, Sales Rep 2, Support Team).
    """
    recipients = target_phones if target_phones is not None else get_team_device_numbers()
    if not recipients:
        return {
            "broadcast_count": 0,
            "recipients": [],
            "note": "No multi-device team phones configured in MULTI_DEVICE_ALERT_PHONES or AGENT_PHONE.",
        }

    results = []
    for phone in recipients:
        res = dispatch_to_customer_handheld(phone, alert_text)
        results.append({"phone": phone, "result": res})

    success_count = sum(1 for r in results if r["result"].get("success"))
    return {
        "broadcast_count": len(recipients),
        "successful_sends": success_count,
        "results": results,
    }


# -------------------------------------------------------------
# 4. REST API Models & Endpoints
# -------------------------------------------------------------
class DeviceCheckRequest(BaseModel):
    phone: str = Field(..., description="Phone number to check")
    country_code: str = Field(default="+91", description="Default country code")


class DeviceDispatchRequest(BaseModel):
    phone: str = Field(..., description="Customer's phone number")
    first_name: Optional[str] = Field(default="Valued Client", description="Customer first name")
    message: Optional[str] = Field(default=None, description="Custom message body to send to device")
    broadcast_to_team: bool = Field(default=False, description="Also broadcast alert to sales team devices")


@router.post("/check-number")
async def api_check_device_number(req: DeviceCheckRequest):
    """Validate whether an entered phone number can receive WhatsApp on a handheld device."""
    e164 = normalize_device_phone(req.phone, req.country_code)
    is_valid = is_valid_mobile_number(e164)
    return {
        "input": req.phone,
        "formatted_e164": e164,
        "is_valid_mobile": is_valid,
        "can_receive_whatsapp": is_valid,
        "device_type": "Handheld Mobile Smartphone",
    }


@router.post("/dispatch")
async def api_dispatch_to_device(req: DeviceDispatchRequest, background_tasks: BackgroundTasks):
    """Send a WhatsApp greeting / questionnaire directly to a customer's physical phone,
    and optionally forward an alert to internal team handheld devices.
    """
    norm_phone = normalize_device_phone(req.phone)
    if not is_valid_mobile_number(norm_phone):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mobile number format ({req.phone}). Please enter a valid 10-digit mobile number.",
        )

    greeting = req.message
    if not greeting:
        name = req.first_name or "Valued Client"
        greeting = (
            f"Hi {name}! 👋 Welcome to *Profilent*.\n\n"
            "We received your property enquiry! Our dedicated advisors are ready to assist you.\n\n"
            f"🗓️ You can also schedule a 1-on-1 consultation directly on your calendar:\n{CALENDLY_URL}\n\n"
            "Feel free to reply here directly on your phone with any preferences (location, budget, BHK)!"
        )

    dispatch_res = await run_in_threadpool(dispatch_to_customer_handheld, norm_phone, greeting)

    if req.broadcast_to_team:
        team_alert = f"📢 *New Customer Active on WhatsApp:*\n• Phone: {norm_phone}\n• Name: {req.first_name}\n• Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        background_tasks.add_task(broadcast_to_multiple_devices, team_alert)

    return {
        "status": "dispatched" if dispatch_res.get("success") else "failed",
        "customer_phone": norm_phone,
        "dispatch_details": dispatch_res,
        "team_broadcast_queued": req.broadcast_to_team,
    }


@router.get("/capabilities")
async def api_device_capabilities():
    """Returns technical details and answers for multi-device WhatsApp architecture."""
    team_phones = get_team_device_numbers()
    sender_digits = re.sub(r"\D", "", TWILIO_WHATSAPP_FROM)
    is_sandbox = sender_digits == "14155238886"

    return {
        "can_send_to_other_numbers": True,
        "can_customer_converse_on_handheld_phone": True,
        "can_send_to_multiple_devices": True,
        "multi_device_mechanisms": {
            "customer_side": "WhatsApp Multi-Device Sync (Automatically mirrors across up to 4 linked companion devices for the customer: phone, tablet, desktop, web).",
            "team_side": f"Configured to broadcast alerts to {len(team_phones)} team handheld devices simultaneously.",
            "team_devices": team_phones,
        },
        "twilio_environment": {
            "sender_number": TWILIO_WHATSAPP_FROM,
            "mode": "Sandbox (Requires recipient to have joined sandbox)" if is_sandbox else "Production WhatsApp Business API (Sends to any mobile globally)",
        },
    }

