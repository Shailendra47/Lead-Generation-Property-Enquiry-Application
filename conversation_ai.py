"""Hybrid Conversational AI Engine powered by Google Gemini for WhatsApp property qualification.

Features:
- Natural, realistic conversational real-estate advisor persona for Profilent.
- Smart slot extraction for all 6 core criteria:
  (location_preference, bhk_preference, budget_range, planning_to_buy, payment_option, remarks_requirements).
- Hybrid UI: Pairs natural AI dialogue with clean quick-reply choices when prompting for
  structured fields (budget, timeline, BHK, payment).
- Google Gemini Integration: Fast, resilient Gemini LLM slot extraction and multi-lingual conversations.
- Full downstream integration: Directly triggers MongoDB Atlas storage, Zoho CRM Leads upsert,
  salesperson notifications, and Calendly appointment booking.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
import time

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from whatsapp_bot import build_prefilled_calendly_url, format_completion_summary_message
from question_manager import (
    get_active_questions_sync,
    format_question_prompt_for_slot,
    invalidate_questions_cache,
)

load_dotenv()

logger = logging.getLogger("uvicorn.error")

# Global reusable session with keep-alive and connection pooling
_ai_session = requests.Session()
_ai_adapter = HTTPAdapter(
    pool_connections=5,
    pool_maxsize=10,
    max_retries=Retry(total=1, backoff_factor=0.2, status_forcelist=[502, 503, 504], raise_on_status=False),
)
_ai_session.mount("https://", _ai_adapter)
_ai_session.mount("http://", _ai_adapter)

_last_env_check = 0.0
_cached_gemini_key = ""
_cached_gemini_model = "gemini-3.1-flash-lite"

PRIMARY_GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

_exhausted_models: dict[str, float] = {}

def _is_model_exhausted(model_name: str) -> bool:
    exp = _exhausted_models.get(model_name)
    if exp and time.monotonic() < exp:
        return True
    return False

def _mark_model_exhausted(model_name: str, duration_sec: float = 600.0) -> None:
    _exhausted_models[model_name] = time.monotonic() + duration_sec
    logger.warning("Gemini model %s marked exhausted / rate-limited for %ds", model_name, int(duration_sec))

def _get_candidate_models(configured_model: str) -> list[str]:
    candidates = []
    if configured_model and not _is_model_exhausted(configured_model):
        candidates.append(configured_model)
    for m in PRIMARY_GEMINI_MODELS:
        if m not in candidates and not _is_model_exhausted(m):
            candidates.append(m)
    return candidates


def _get_gemini_config() -> tuple[str, str]:
    """Retrieve dynamic Gemini settings from environment with low-overhead caching."""
    global _last_env_check, _cached_gemini_key, _cached_gemini_model
    now = time.monotonic()
    if now - _last_env_check > 5.0 or not _cached_gemini_key:
        load_dotenv(override=False)
        _cached_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        raw_model = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip() or "gemini-3.1-flash-lite"
        _cached_gemini_model = raw_model
        _last_env_check = now
    return _cached_gemini_key, _cached_gemini_model


def is_ai_configured() -> bool:
    """Check if Google Gemini API key is present."""
    g_key, _ = _get_gemini_config()
    return bool(g_key and not g_key.startswith("replace_with"))


STANDARD_OPTIONS = {
    "location_preference": [
        "Haldwani (Rampur Rd, Kaladhungi Rd, Bithoria, Mukhani)",
        "Nainital / Bhimtal",
        "Almora",
        "Other (please type the area)",
    ],
    "planning_to_buy": [
        "Immediately",
        "Within 1 Month",
        "Within 3 Months",
        "Within 6 Months",
        "Within 12 Months",
        "More Than 1 Year",
        "Just Exploring / Researching",
    ],
    "budget_range": [
        "Up to 25 lakh",
        "25 lakh - 50 lakh",
        "50 lakh - 1 crore",
        "1 crore - 2 crore",
        "Above 2 crore",
    ],
    "payment_option": [
        "Bank Finance with EMI",
        "Full Cash Payment",
        "Home Loan (Mortgage)",
        "Down Payment Plan",
        "Construction-Linked Payment Plan (CLP)",
        "Possession-Linked Payment Plan (PLP)",
        "Installment Payment Plan",
        "Developer Financing",
        "Rent-to-Own Scheme",
        "Part Cash + Home Loan",
        "Part Payment with Balance at Registration",
        "Flexible Payment Plan",
        "Subvention Scheme (Developer Pays Pre-EMI)",
        "Joint Home Loan (Co-applicant Financing)",
        "NRI Payment Plan (for Overseas Buyers)",
    ],
    "bhk_preference": [
        "1 RK",
        "1 BHK",
        "2 BHK",
        "2.5 BHK",
        "3 BHK",
        "3.5 BHK",
        "4 BHK",
        "5 BHK+",
        "Villa",
        "Penthouse",
    ],
}

QUESTION_PROMPTS_MULTILINGUAL = {
    "en": {
        "location_preference": (
            "Which area are you interested in?\n"
            "1. Haldwani (Rampur Rd, Kaladhungi Rd, Bithoria, Mukhani)\n"
            "2. Nainital / Bhimtal\n"
            "3. Almora\n"
            "4. Other (please type the city or locality)"
        ),
        "planning_to_buy": (
            "When are you planning to buy?\n"
            "1. Immediately\n"
            "2. Within 1 Month\n"
            "3. Within 3 Months\n"
            "4. Within 6 Months\n"
            "5. Within 12 Months\n"
            "6. More Than 1 Year\n"
            "7. Just Exploring / Researching"
        ),
        "budget_range": (
            "What is your budget range?\n"
            "1. Up to 25 lakh\n"
            "2. 25 lakh - 50 lakh\n"
            "3. 50 lakh - 1 crore\n"
            "4. 1 crore - 2 crore\n"
            "5. Above 2 crore"
        ),
        "payment_option": (
            "How do you plan to pay?\n"
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
        "bhk_preference": (
            "Which configuration / BHK do you need?\n"
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
        "remarks_requirements": (
            "Do you have any specific remarks or requirements (such as parking, floor preference, facing direction, hill view, or nearby schools)? Type whatever you need, or type \"skip\" if none."
        ),
        "skip_warning": "Please answer this question before we move on:\n\n",
        "inappropriate": (
            "I am here to assist you exclusively with finding the best properties and real estate opportunities in Haldwani & Uttarakhand. 🏡\n\n"
            "Let's focus on your property search!\n\n"
        ),
    },
    "hinglish": {
        "location_preference": (
            "Aap kis area me property dekh rahe hain?\n"
            "1. Haldwani (Rampur Rd, Kaladhungi Rd, Bithoria, Mukhani)\n"
            "2. Nainital / Bhimtal\n"
            "3. Almora\n"
            "4. Other (apna area type karein)"
        ),
        "planning_to_buy": (
            "Aapka timeline kya hai (kab tak lena chahte hain)?\n"
            "1. Immediately\n"
            "2. Within 1 Month\n"
            "3. Within 3 Months\n"
            "4. Within 6 Months\n"
            "5. Within 12 Months\n"
            "6. More Than 1 Year\n"
            "7. Just Exploring / Researching"
        ),
        "budget_range": (
            "Aapka budget kitna hai?\n"
            "1. Up to 25 lakh\n"
            "2. 25 lakh - 50 lakh\n"
            "3. 50 lakh - 1 crore\n"
            "4. 1 crore - 2 crore\n"
            "5. Above 2 crore"
        ),
        "payment_option": (
            "Aap payment kaise plan kar rahe hain?\n"
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
        "bhk_preference": (
            "Aapko kitne BHK / type ki requirement hai?\n"
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
        "remarks_requirements": (
            "Aapki koi bhi special requirement ya remarks (jaise parking, floor, facing, garden, schools ke paas)? Jo bhi requirement ho yahan likhein, ya \"skip\" type karein."
        ),
        "skip_warning": "Aage badhne se pehle please is sawaal ka jawab dein:\n\n",
        "inappropriate": (
            "Main yahan Haldwani aur Uttarakhand me best property dhoondhne me aapki madad karne ke liye hoon. 🏡\n\n"
            "Aaiye aapki property search pe focus karte hain!\n\n"
        ),
    },
    "hi": {
        "location_preference": (
            "आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
            "1. हल्द्वानी (रामपुर रोड, कालाढूंगी रोड, बिठौरिया, मुखानी)\n"
            "2. नैनीताल / भीमताल\n"
            "3. अल्मोड़ा\n"
            "4. अन्य (कृपया इलाके का नाम लिखें)"
        ),
        "planning_to_buy": (
            "आपकी समयसीमा क्या है (कब तक प्रॉपर्टी लेना चाहते हैं)?\n"
            "1. Immediately (तुरंत)\n"
            "2. Within 1 Month (1 महीने के भीतर)\n"
            "3. Within 3 Months (1 से 3 महीने में)\n"
            "4. Within 6 Months (3 से 6 महीने में)\n"
            "5. Within 12 Months (12 महीने के भीतर)\n"
            "6. More Than 1 Year (1 वर्ष से अधिक)\n"
            "7. Just Exploring / Researching (अभी सिर्फ जानकारी ले रहे हैं)"
        ),
        "budget_range": (
            "आपका बजट रेंज कितना है?\n"
            "1. Up to 25 lakh (25 लाख तक)\n"
            "2. 25 lakh - 50 lakh (25 लाख से 50 लाख)\n"
            "3. 50 lakh - 1 crore (50 लाख से 1 करोड़)\n"
            "4. 1 crore - 2 crore (1 करोड़ से 2 करोड़)\n"
            "5. Above 2 crore (2 करोड़ से अधिक)"
        ),
        "payment_option": (
            "आप भुगतान किस प्रकार करने की योजना बना रहे हैं?\n"
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
        "bhk_preference": (
            "आपको किस प्रकार / BHK की आवश्यकता है?\n"
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
        "remarks_requirements": (
            "क्या आपकी कोई विशेष आवश्यकता या टिप्पणी (जैसे पार्किंग, फ्लोर, दिशा, गार्डन या स्कूल के पास) है? अपनी आवश्यकता यहां लिखें, अथवा \"skip\" टाइप करें।"
        ),
        "skip_warning": "आगे बढ़ने से पहले कृपया इस प्रश्न का उत्तर दें:\n\n",
        "inappropriate": (
            "मैं यहां हल्द्वानी और उत्तराखंड में बेहतरीन प्रॉपर्टी ढूंढने में आपकी सहायता के लिए हूं। 🏡\n\n"
            "आइए आपकी प्रॉपर्टी खोज पर ध्यान केंद्रित करते हैं!\n\n"
        ),
    },
}

QUESTION_PROMPTS = QUESTION_PROMPTS_MULTILINGUAL["en"]

DEFAULT_REQUIRED_SLOTS = [
    "location_preference",
    "planning_to_buy",
    "budget_range",
    "payment_option",
    "bhk_preference",
]


class _DynamicRequiredSlots(list):
    """Dynamic list proxy that returns active required question slot keys configured by Admin."""

    def _get_slots(self) -> list[str]:
        try:
            qs = get_active_questions_sync()
            req = [q["slot_key"] for q in qs if q.get("is_required", True) and q.get("slot_key")]
            return req if req else DEFAULT_REQUIRED_SLOTS
        except Exception:
            return DEFAULT_REQUIRED_SLOTS

    def __iter__(self):
        return iter(self._get_slots())

    def __contains__(self, item):
        return item in self._get_slots()

    def __len__(self):
        return len(self._get_slots())

    def __getitem__(self, idx):
        return self._get_slots()[idx]

    def __bool__(self):
        return bool(self._get_slots())

    def __repr__(self):
        return repr(self._get_slots())


ORDERED_REQUIRED_SLOTS = _DynamicRequiredSlots()

_SPEAK_TO_HUMAN_PATTERN = re.compile(
    r"\b(speak to (?:a )?(?:person|human|executive|agent|representative|someone)|"
    r"talk to (?:a )?(?:person|human|executive|agent|representative|someone)|"
    r"call me|human agent|sales executive|connect (?:me )?to (?:a )?(?:person|human|executive|agent)|"
    r"human|person chahiye|agent se baat|executive se baat|baat karni hai|call karo|insan se baat|"
    r"बात करनी है|कॉल करो|कॉल करें|एजेंट से बात|इंसान से बात|कॉल कीजिए)\b",
    re.IGNORECASE,
)

_HINGLISH_WORDS = re.compile(
    r"\b(mujhe|chahiye|aap|aapka|aapko|hai|hain|kya|bhai|karna|hoga|batao|bataiye|"
    r"kitna|kitne|mein?|se|ke|liye|ka|ki|ko|nahi|nahin|dhanyawad|ghar|makan|zameen|"
    r"raha|rahe|karein|bhi|aur|kaun|kaunsa|kab|tak|dekh|baat|karo|shukriya|namaste|namaskar)\b",
    re.IGNORECASE,
)

_HINDI_NUMS = {
    "१": 1, "२": 2, "३": 3, "४": 4, "५": 5,
    "६": 6, "७": 7, "८": 8, "९": 9, "१०": 10,
    "११": 11, "१२": 12, "१३": 13, "१४": 14, "१५": 15,
}


def detect_language(text: str, current_lang: str = "en") -> str:
    """Strictly English only - Hindi and Hinglish disabled throughout the project."""
    return "en"


def get_question_prompt(slot: str, lang: str = "en") -> str:
    """Retrieve localized question prompt for the given slot, checking dynamic admin questions first."""
    try:
        qs = get_active_questions_sync()
        for q in qs:
            if q.get("slot_key") == slot:
                return format_question_prompt_for_slot(q, lang=lang)
    except Exception:
        pass
    prompts = QUESTION_PROMPTS_MULTILINGUAL.get(lang, QUESTION_PROMPTS_MULTILINGUAL["en"])
    return prompts.get(slot, QUESTION_PROMPTS_MULTILINGUAL["en"].get(slot, ""))


LOCALITY_SUGGESTIONS = {
    "haldwani": {
        "title": "Haldwani",
        "intro": {
            "en": "Haldwani is a wonderful and rapidly growing city with excellent healthcare, premier schools, and smooth highway connectivity! 🌟",
            "hi": "हल्द्वानी बेहतरीन कनेक्टिविटी, उत्तम स्वास्थ्य सेवाओं और स्कूलों वाला तेजी से विकसित होता शहर है! 🌟",
            "hinglish": "Haldwani ek rapidly growing hub hai with great road connectivity, top schools, aur medical facilities! 🌟",
        },
        "question": {
            "en": (
                "Which specific area in Haldwani do you prefer?\n"
                "1. Rampur Road (High connectivity & commercial access)\n"
                "2. Kaladhungi Road (Peaceful residential surroundings)\n"
                "3. Bithoria / Mukhani (Prime family neighborhoods)\n"
                "4. Kusumkhera / Kathgodam (Near railway & foothills)\n"
                "(Or reply with any specific colony/locality you prefer)"
            ),
            "hi": (
                "हल्द्वानी में आप किस खास इलाके को प्राथमिकता देते हैं?\n"
                "1. रामपुर रोड (उत्कृष्ट कनेक्टिविटी)\n"
                "2. कालाढूंगी रोड (शांत आवासीय माहौल)\n"
                "3. बिठौरिया / मुखानी (परिवारों की पहली पसंद)\n"
                "4. कुसुमखेड़ा / काठगोदाम (स्टेशन व तलहटी के समीप)\n"
                "(अथवा अपनी पसंद का कोई भी इलाका लिखें)"
            ),
            "hinglish": (
                "Haldwani me aap kis specific area ko prefer karte hain?\n"
                "1. Rampur Road (Prime connectivity & markets)\n"
                "2. Kaladhungi Road (Peaceful residential area)\n"
                "3. Bithoria / Mukhani (Popular family neighborhood)\n"
                "4. Kusumkhera / Kathgodam (Near station & foothills)\n"
                "(Ya koi bhi specific area/colony type karein)"
            ),
        },
        "options": [
            "Rampur Road",
            "Kaladhungi Road",
            "Bithoria / Mukhani",
            "Kusumkhera / Kathgodam",
        ],
    },
    "nainital": {
        "title": "Nainital / Bhimtal",
        "intro": {
            "en": "Nainital & Bhimtal offer refreshing weather and scenic Himalayan vistas all year round—ideal for peaceful living or holiday homes! 🌄",
            "hi": "नैनीताल और भीमताल साल भर सुहावने मौसम और खूबसूरत झीलों के लिए प्रसिद्ध हैं—शांतिपूर्ण जीवन और हॉलिडे होम हेतु सर्वोत्तम! 🌄",
            "hinglish": "Nainital aur Bhimtal serene weather aur beautiful lake views ke liye best hain—perfect for holiday homes! 🌄",
        },
        "question": {
            "en": (
                "Which specific area in Nainital / Bhimtal do you prefer?\n"
                "1. Bhimtal Lake area\n"
                "2. Bhowali (Gateway to hills & clean climate)\n"
                "3. Naukuchiatal / Sattal (Serene lake surroundings)\n"
                "4. Nainital Mallital / Tallital\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "नैनीताल / भीमताल में आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. भीमताल झील क्षेत्र\n"
                "2. भवाली (स्वच्छ आबोहवा व कनेक्टिविटी)\n"
                "3. नौकुचियाताल / सातताल (शांत प्राकृतिक क्षेत्र)\n"
                "4. नैनीताल मल्लीताल / तल्लीताल\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Nainital / Bhimtal me aap kis area ko prefer karte hain?\n"
                "1. Bhimtal Lake area\n"
                "2. Bhowali\n"
                "3. Naukuchiatal / Sattal\n"
                "4. Nainital Mallital / Tallital\n"
                "(Ya koi bhi specific locality type karein)"
            ),
        },
        "options": [
            "Bhimtal Lake area",
            "Bhowali",
            "Naukuchiatal / Sattal",
            "Nainital Mallital / Tallital",
        ],
    },
    "almora": {
        "title": "Almora",
        "intro": {
            "en": "Almora offers a serene retreat lifestyle with panoramic snow-capped Himalayan views and rich cultural heritage! 🏔️",
            "hi": "अल्मोड़ा हिमालय के मनोहारी दृश्यों, समृद्ध संस्कृति और शांत वातावरण के लिए विख्यात है! 🏔️",
            "hinglish": "Almora peaceful mountain living aur majestic Himalayan views ke liye famous hai! 🏔️",
        },
        "question": {
            "en": (
                "Which specific locality in/around Almora do you prefer?\n"
                "1. Bright End Corner / Kasar Devi\n"
                "2. Ranikhet Road\n"
                "3. Dharanaula / Almora Town\n"
                "4. Hawalbagh / Kosi Valley\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "अल्मोड़ा में आप किस क्षेत्र को प्राथमिकता देते हैं?\n"
                "1. ब्राइट एंड कॉर्नर / कसार देवी\n"
                "2. रानीखेत रोड\n"
                "3. धारानौला / अल्मोड़ा टाउन\n"
                "4. हवालबाग / कोसी वैली\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Almora me aap kis area ko prefer karte hain?\n"
                "1. Bright End Corner / Kasar Devi\n"
                "2. Ranikhet Road\n"
                "3. Dharanaula / Almora Town\n"
                "4. Hawalbagh / Kosi Valley\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Bright End Corner / Kasar Devi",
            "Ranikhet Road",
            "Dharanaula / Almora Town",
            "Hawalbagh / Kosi Valley",
        ],
    },
    "bhimtal": {
        "title": "Bhimtal",
        "intro": {
            "en": "Bhimtal is known for its beautiful serene lake, pleasant climate, and prime holiday-home opportunities! 🌊",
            "hi": "भीमताल अपनी शांत झील, साल भर सुहावने मौसम और बेहतरीन हॉलिडे होम निवेश के लिए जाना जाता है! 🌊",
            "hinglish": "Bhimtal serene lake views, pleasant weather aur prime holiday-home investment ke liye best hai! 🌊",
        },
        "question": {
            "en": (
                "Which specific area in Bhimtal do you prefer?\n"
                "1. Bhimtal Lake Promenade / Mehragaon\n"
                "2. Bhowali-Bhimtal Road\n"
                "3. Naukuchiatal Road / Sattal\n"
                "4. Gorakhal Road / Ghorakhal\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "भीमताल में आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. भीमताल लेक प्रोमेनाड / मेहरागांव\n"
                "2. भवाली-भीमताल रोड\n"
                "3. नौकुचियाताल रोड / सातताल\n"
                "4. घोड़ाखाल रोड\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Bhimtal me aap kis area ko prefer karte hain?\n"
                "1. Bhimtal Lake Promenade / Mehragaon\n"
                "2. Bhowali-Bhimtal Road\n"
                "3. Naukuchiatal Road / Sattal\n"
                "4. Gorakhal Road / Ghorakhal\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Bhimtal Lake Promenade / Mehragaon",
            "Bhowali-Bhimtal Road",
            "Naukuchiatal Road / Sattal",
            "Gorakhal Road / Ghorakhal",
        ],
    },
    "bhowali": {
        "title": "Bhowali",
        "intro": {
            "en": "Bhowali is the scenic fruit market gateway to the hills, blessed with clean pine air and close to Kainchi Dham! 🌲",
            "hi": "भवाली शुद्ध चीड़ की हवा, कैंची धाम के समीप और नैनीताल-अल्मोड़ा जंक्शन पर बसा एक सुंदर हिल स्टेशन है! 🌲",
            "hinglish": "Bhowali clean pine air, Kainchi Dham proximity aur central connectivity ke liye famous hai! 🌲",
        },
        "question": {
            "en": (
                "Which specific area in/around Bhowali do you prefer?\n"
                "1. Kainchi Dham Road / Sanatorium\n"
                "2. Ramgarh Road / Niglat\n"
                "3. Bhowali-Bhimtal Road\n"
                "4. Nainital Road / Bhowali Main Market\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "भवाली में आप किस क्षेत्र को प्राथमिकता देते हैं?\n"
                "1. कैंची धाम रोड / सेनेटोरियम\n"
                "2. रामगढ़ रोड / निगलात\n"
                "3. भवाली-भीमताल रोड\n"
                "4. नैनीताल रोड / भवाली मुख्य बाजार\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Bhowali me aap kis area ko prefer karte hain?\n"
                "1. Kainchi Dham Road / Sanatorium\n"
                "2. Ramgarh Road / Niglat\n"
                "3. Bhowali-Bhimtal Road\n"
                "4. Nainital Road / Bhowali Main Market\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Kainchi Dham Road / Sanatorium",
            "Ramgarh Road / Niglat",
            "Bhowali-Bhimtal Road",
            "Nainital Road / Bhowali Main Market",
        ],
    },
    "dehradun": {
        "title": "Dehradun",
        "intro": {
            "en": "Dehradun is the vibrant capital city in the Doon Valley, renowned for premier institutions, pleasant weather, and rapid urban growth! 🌟",
            "hi": "देहरादून दून घाटी का जीवंत शहर है, जो प्रतिष्ठित संस्थानों, सुहावने मौसम और आधुनिक विकास के लिए प्रसिद्ध है! 🌟",
            "hinglish": "Dehradun premier schools, pleasant Doon valley climate aur rapid development ke liye top choice hai! 🌟",
        },
        "question": {
            "en": (
                "Which specific area in Dehradun do you prefer?\n"
                "1. Rajpur Road / Sahastradhara Road (Prime & upscale living)\n"
                "2. GMS Road / Ballupur / Vasant Vihar (Central & residential)\n"
                "3. Haridwar Bypass / ISBT area (High highway connectivity)\n"
                "4. Clement Town / Subhash Nagar (Peaceful institutional zone)\n"
                "(Or reply with any specific colony/locality you prefer)"
            ),
            "hi": (
                "देहरादून में आप किस खास इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. राजपुर रोड / सहस्रधारा रोड (प्रीमियम आवासीय क्षेत्र)\n"
                "2. जीएमएस रोड / बल्लूपुर / वसंत विहार (केंद्रीय आवासीय इलाका)\n"
                "3. हरिद्वार बाईपास / आईएसबीटी (उत्कृष्ट कनेक्टिविटी)\n"
                "4. क्लेमेंट टाउन / सुभाष नगर (शांत व सुव्यवस्थित)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Dehradun me aap kis specific area ko prefer karte hain?\n"
                "1. Rajpur Road / Sahastradhara Road (Prime & upscale)\n"
                "2. GMS Road / Ballupur / Vasant Vihar (Central family hubs)\n"
                "3. Haridwar Bypass / ISBT area (Great connectivity)\n"
                "4. Clement Town / Subhash Nagar (Peaceful residential)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Rajpur Road / Sahastradhara Road",
            "GMS Road / Ballupur / Vasant Vihar",
            "Haridwar Bypass / ISBT area",
            "Clement Town / Subhash Nagar",
        ],
    },
    "rishikesh": {
        "title": "Rishikesh",
        "intro": {
            "en": "Rishikesh is the world-renowned yoga capital on the banks of the Holy Ganga, offering peaceful living and spiritual vibrancy! 🧘",
            "hi": "ऋषिकेश मां गंगा के तट पर स्थित विश्व विख्यात योग नगरी है, जो आध्यात्मिक शांति और प्राकृतिक सौंदर्य का अनुपम संगम है! 🧘",
            "hinglish": "Rishikesh Holy Ganga ke bank pe spiritual peace, yoga vibe aur high-yield holiday rentals ke liye perfect hai! 🧘",
        },
        "question": {
            "en": (
                "Which specific area in Rishikesh do you prefer?\n"
                "1. Tapovan / Laxman Jhula (Tourist & yoga hub, high rental returns)\n"
                "2. Bairaj Road / Pashulok / AIIMS area (Prime residential & medical hub)\n"
                "3. Haridwar-Rishikesh Highway / Shyampur (Easy transit & upcoming projects)\n"
                "4. Dhalwala / Muni Ki Reti (River proximity & settled community)\n"
                "(Or reply with any specific colony/locality you prefer)"
            ),
            "hi": (
                "ऋषिकेश में आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. तपोवन / लक्ष्मण झूला (योग व टूरिस्ट हब, हाई रेंटल यील्ड)\n"
                "2. बैराज रोड / पशुलोक / एम्स क्षेत्र (प्रीमियम आवासीय क्षेत्र)\n"
                "3. हरिद्वार-ऋषिकेश हाईवे / श्यामपुर (उत्कृष्ट कनेक्टिविटी)\n"
                "4. ढालवाला / मुनि की रेती (गंगा के समीप शांत इलाका)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Rishikesh me aap kis area ko prefer karte hain?\n"
                "1. Tapovan / Laxman Jhula (Prime tourism & rental hub)\n"
                "2. Bairaj Road / Pashulok / AIIMS area (Family residential)\n"
                "3. Haridwar-Rishikesh Highway / Shyampur (Fast connectivity)\n"
                "4. Dhalwala / Muni Ki Reti (Close to Ganga ghats)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Tapovan / Laxman Jhula",
            "Bairaj Road / Pashulok / AIIMS area",
            "Haridwar-Rishikesh Highway / Shyampur",
            "Dhalwala / Muni Ki Reti",
        ],
    },
    "haridwar": {
        "title": "Haridwar",
        "intro": {
            "en": "Haridwar is the sacred gateway of the gods, blending sacred heritage with prominent industrial and highway infrastructure! 🕉️",
            "hi": "हरिद्वार पावन गंगा जी का पावन धाम है, जहाँ धार्मिक आस्था और आधुनिक औद्योगिक बुनियादी ढांचा साथ-साथ चलते हैं! 🕉️",
            "hinglish": "Haridwar sacred Ganga heritage ke saath top highway connectivity aur industrial growth offer karta hai! 🕉️",
        },
        "question": {
            "en": (
                "Which specific area in Haridwar do you prefer?\n"
                "1. Shivalik Nagar / BHEL area (Planned township living)\n"
                "2. Delhi-Haridwar Highway / Bahadrabad (Industrial & high connectivity)\n"
                "3. Jwalapur / Ranipur (Commercial & settled residential)\n"
                "4. Kankhal / Ganga Ghats (Cultural heritage & riverfront)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "हरिद्वार में आप किस क्षेत्र को प्राथमिकता देते हैं?\n"
                "1. शिवालिक नगर / भेल क्षेत्र (योजनाबद्ध टाउनशिप)\n"
                "2. दिल्ली-हरिद्वार हाईवे / बहादराबाद (हाईवे कनेक्टिविटी)\n"
                "3. ज्वालापुर / रानीपुर (सुस्थापित आवासीय व व्यावसायिक)\n"
                "4. कनखल / गंगा घाट क्षेत्र (सांस्कृतिक व धार्मिक वातावरण)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Haridwar me aap kis area ko prefer karte hain?\n"
                "1. Shivalik Nagar / BHEL area (Planned township)\n"
                "2. Delhi-Haridwar Highway / Bahadrabad (Prime connectivity)\n"
                "3. Jwalapur / Ranipur (Settled residential)\n"
                "4. Kankhal / Ganga Ghats (Heritage & riverfront)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Shivalik Nagar / BHEL area",
            "Delhi-Haridwar Highway / Bahadrabad",
            "Jwalapur / Ranipur",
            "Kankhal / Ganga Ghats",
        ],
    },
    "mussoorie": {
        "title": "Mussoorie",
        "intro": {
            "en": "Mussoorie, the Queen of Hills, is famous for colonial heritage, misty valleys, and prestigious holiday estates! 🏔️",
            "hi": "पहाड़ों की रानी मसूरी अपने मनोरम नजारों, ठंडी वादियों और प्रतिष्ठित हॉलिडे होम्स के लिए जानी जाती है! 🏔️",
            "hinglish": "Queen of the Hills Mussoorie scenic valley views aur luxury holiday-home living ke liye world-famous hai! 🏔️",
        },
        "question": {
            "en": (
                "Which specific area in Mussoorie do you prefer?\n"
                "1. Mall Road / Kulri (Central shopping & lifestyle)\n"
                "2. Landour / Char Dukan (Serene, heritage pine forests)\n"
                "3. Barlowganj / Jharipani (Suburban foothill breeze)\n"
                "4. Kempty Road / Library Chowk (Scenic valley stretches)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "मसूरी में आप किस क्षेत्र को प्राथमिकता देते हैं?\n"
                "1. माल रोड / कुलड़ी (केंद्रीय जीवनशैली)\n"
                "2. लंढौर / चार दुकान (शांत चीड़ के जंगल व हेरिटेज)\n"
                "3. बारलोगंज / झड़ीपानी (शांत आवासीय वातावरण)\n"
                "4. कैंपटी रोड / लाइब्रेरी चौक (मनोरम घाटी दृश्य)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Mussoorie me aap kis area ko prefer karte hain?\n"
                "1. Mall Road / Kulri (Central hub)\n"
                "2. Landour / Char Dukan (Peaceful pine forest retreat)\n"
                "3. Barlowganj / Jharipani (Serene residential)\n"
                "4. Kempty Road / Library Chowk (Valley views)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Mall Road / Kulri",
            "Landour / Char Dukan",
            "Barlowganj / Jharipani",
            "Kempty Road / Library Chowk",
        ],
    },
    "mukteshwar": {
        "title": "Mukteshwar",
        "intro": {
            "en": "Mukteshwar boasts unmatched 180-degree Himalayan snow views, apple orchards, and pristine mountain calm! 🍎",
            "hi": "मुक्तेश्वर हिमालय की बर्फीली चोटियों के 180-डिग्री मनमोहक दृश्यों, सेब के बागों और असीम शांति के लिए प्रसिद्ध है! 🍎",
            "hinglish": "Mukteshwar unmatched Himalayan snow-peak views, apple orchards aur peaceful mountain cottages ke liye top spot hai! 🍎",
        },
        "question": {
            "en": (
                "Which specific area in/around Mukteshwar do you prefer?\n"
                "1. Sargakhet / Chauli Ki Jali (Panoramic cliff & peak views)\n"
                "2. Dhanachuli / Bhatelia (Fruit belt & boutique stays)\n"
                "3. Sitla / Sunkiya (Dense deodar forests & birding)\n"
                "4. Ramgarh Road / Fruit Valley (Serene valley retreats)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "मुक्तेश्वर में आप किस इलाके को प्राथमिकता देते हैं?\n"
                "1. सरगाखेत / चौली की जाली (भव्य हिमालयी दृश्य)\n"
                "2. धानाचूली / भटेलिया (ऑर्चर्ड व बुटीक विला)\n"
                "3. सीतला / सुंकिया (देवदार के घने जंगल)\n"
                "4. रामगढ़ रोड / फ्रूट वैली (शांत घाटी परिवेश)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Mukteshwar me aap kis area ko prefer karte hain?\n"
                "1. Sargakhet / Chauli Ki Jali (Himalayan peak views)\n"
                "2. Dhanachuli / Bhatelia (Orchards & cottage belt)\n"
                "3. Sitla / Sunkiya (Quiet deodar forest living)\n"
                "4. Ramgarh Road / Fruit Valley (Valley retreat)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Sargakhet / Chauli Ki Jali",
            "Dhanachuli / Bhatelia",
            "Sitla / Sunkiya",
            "Ramgarh Road / Fruit Valley",
        ],
    },
    "ranikhet": {
        "title": "Ranikhet",
        "intro": {
            "en": "Ranikhet is an idyllic hill station featuring lush golf courses, pine woods, and panoramic views of Western peaks! ⛳",
            "hi": "रानीखेत हरे-भरे गोल्फ कोर्स, देवदार के जंगलों और हिमालय की विशाल चोटियों के दृश्यों के लिए प्रसिद्ध है! ⛳",
            "hinglish": "Ranikhet lush golf grounds, pine trails aur majestic Himalayan vistas ke liye popular holiday home hub hai! ⛳",
        },
        "question": {
            "en": (
                "Which specific area in Ranikhet do you prefer?\n"
                "1. Majkhali / Golf Course area (Prime luxury cottages & open views)\n"
                "2. Chaubatia / Kalika (Lush orchards & temple area)\n"
                "3. Ranikhet Cantonment / Mall Road (Pristine, clean environment)\n"
                "4. Tarikhet / Almora Road (Upcoming investment belt)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "रानीखेत में आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. मजखाली / गोल्फ कोर्स (प्रीमियम कॉटेज व खुला दृश्य)\n"
                "2. चौबटिया / कालिका (फल उद्यान व मंदिर क्षेत्र)\n"
                "3. रानीखेत छावनी / माल रोड (स्वच्छ व शांत वातावरण)\n"
                "4. ताड़ीखेत / अल्मोड़ा रोड (तेजी से उभरता क्षेत्र)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Ranikhet me aap kis area ko prefer karte hain?\n"
                "1. Majkhali / Golf Course area (Prime mountain cottages)\n"
                "2. Chaubatia / Kalika (Fruit orchards & nature)\n"
                "3. Ranikhet Cantonment / Mall Road (Peaceful & pristine)\n"
                "4. Tarikhet / Almora Road (Great connectivity)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Majkhali / Golf Course area",
            "Chaubatia / Kalika",
            "Ranikhet Cantonment / Mall Road",
            "Tarikhet / Almora Road",
        ],
    },
    "ramnagar": {
        "title": "Ramnagar / Jim Corbett",
        "intro": {
            "en": "Ramnagar is the premier gateway to Jim Corbett National Park, famed for luxury eco-resorts and jungle retreats! 🐯",
            "hi": "रामनगर विश्व प्रसिद्ध जिम कॉर्बेट नेशनल पार्क का मुख्य द्वार है, जो इको-रिसॉर्ट्स और जंगल जीवनशैली के लिए विख्यात है! 🐯",
            "hinglish": "Ramnagar Jim Corbett tiger reserve ka gateway hai, famous for riverfront eco-resorts aur high tourist footfall! 🐯",
        },
        "question": {
            "en": (
                "Which specific area in Ramnagar / Corbett do you prefer?\n"
                "1. Dhikuli / Riverfront Belt (Prime resort & luxury holiday zone)\n"
                "2. Dhela / Jhirna Road (Peaceful jungle buffer zone)\n"
                "3. Ringora / Bailparao (Green forest highway corridor)\n"
                "4. Ramnagar Town / Station Road (Urban conveniences & markets)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "रामनगर / कॉर्बेट में आप किस क्षेत्र को प्राथमिकता देते हैं?\n"
                "1. ढिकुली / रिवरफ्रंट बेल्ट (प्रीमियम रिसॉर्ट व हॉलिडे होम)\n"
                "2. ढेला / झिरना रोड (शांत जंगल बफर क्षेत्र)\n"
                "3. रिंगोड़ा / बैलपड़ाव (हरित वन कॉरिडोर)\n"
                "4. रामनगर टाउन / स्टेशन रोड (शहरी सुविधाएं व बाजार)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Ramnagar / Corbett me aap kis area ko prefer karte hain?\n"
                "1. Dhikuli / Riverfront Belt (Luxury resort & villa strip)\n"
                "2. Dhela / Jhirna Road (Peaceful nature buffer)\n"
                "3. Ringora / Bailparao (Green highway belt)\n"
                "4. Ramnagar Town / Station Road (Central residential)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Dhikuli / Riverfront Belt",
            "Dhela / Jhirna Road",
            "Ringora / Bailparao",
            "Ramnagar Town / Station Road",
        ],
    },
    "rudrapur": {
        "title": "Rudrapur",
        "intro": {
            "en": "Rudrapur is the booming industrial and commercial engine of Kumaon with modern residential townships! 🏭",
            "hi": "रुद्रपुर कुमाऊं का प्रमुख औद्योगिक व व्यापारिक केंद्र है, जहाँ आधुनिक टाउनशिप्स का तेजी से विस्तार हो रहा है! 🏭",
            "hinglish": "Rudrapur industrial hub SIDCUL, modern gated societies aur high rental demand ke liye famous hai! 🏭",
        },
        "question": {
            "en": (
                "Which specific area in Rudrapur do you prefer?\n"
                "1. Awas Vikas / Civil Lines (Established central residential)\n"
                "2. Kichha Road / SIDCUL Area (Near industrial parks & jobs)\n"
                "3. Delhi-Nainital Highway / Metropolis (Modern gated societies & malls)\n"
                "4. Kashipur Bypass / Dineshpur Road (Developing investment plots)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "रुद्रपुर में आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. आवास विकास / सिविल लाइंस (सुव्यवस्थित केंद्रीय इलाका)\n"
                "2. किच्छा रोड / सिडकुल क्षेत्र (इंडस्ट्रियल हब के समीप)\n"
                "3. दिल्ली-नैनीताल हाईवे / मेट्रोपोलिस (मॉडर्न टाउनशिप)\n"
                "4. काशीपुर बाईपास / दिनेशपुर रोड (किफायती प्लॉट्स)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Rudrapur me aap kis area ko prefer karte hain?\n"
                "1. Awas Vikas / Civil Lines (Central residential)\n"
                "2. Kichha Road / SIDCUL Area (Close to workplace)\n"
                "3. Delhi-Nainital Highway / Metropolis (Modern societies)\n"
                "4. Kashipur Bypass / Dineshpur Road (Investment plots)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Awas Vikas / Civil Lines",
            "Kichha Road / SIDCUL Area",
            "Delhi-Nainital Highway / Metropolis",
            "Kashipur Bypass / Dineshpur Road",
        ],
    },
    "kashipur": {
        "title": "Kashipur",
        "intro": {
            "en": "Kashipur is a historic industrial and educational hub with seamless connectivity to Delhi-NCR! 🛣️",
            "hi": "काशीपुर कुमाऊं का समृद्ध औद्योगिक व शैक्षिक केंद्र है, जो दिल्ली-एनसीआर से सीधी कनेक्टिविटी रखता है! 🛣️",
            "hinglish": "Kashipur booming industry, premier colleges aur Delhi-NCR connectivity ke liye ideal choice hai! 🛣️",
        },
        "question": {
            "en": (
                "Which specific area in Kashipur do you prefer?\n"
                "1. Ramnagar Road / Moradabad Road (Prime commercial & highway belt)\n"
                "2. Aliganj Road / Bazpur Road (Established family residential)\n"
                "3. Railway Colony / Station Road (Central amenities & schools)\n"
                "4. Cheema Square / Awas Vikas (Modern residential colonies)\n"
                "(Or reply with any specific locality you prefer)"
            ),
            "hi": (
                "काशीपुर में आप किस इलाके में प्रॉपर्टी देख रहे हैं?\n"
                "1. रामनगर रोड / मुरादाबाद रोड (प्रीमियम हाईवे कॉरिडोर)\n"
                "2. अलीगंज रोड / बाजपुर रोड (सुव्यवस्थित आवासीय इलाका)\n"
                "3. रेलवे कॉलोनी / स्टेशन रोड (केंद्रीय सुविधाएं)\n"
                "4. चीमा चौराहा / आवास विकास (आधुनिक कॉलोनियां)\n"
                "(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                "Kashipur me aap kis area ko prefer karte hain?\n"
                "1. Ramnagar Road / Moradabad Road (Prime highway belt)\n"
                "2. Aliganj Road / Bazpur Road (Settled family area)\n"
                "3. Railway Colony / Station Road (Central location)\n"
                "4. Cheema Square / Awas Vikas (Modern colonies)\n"
                "(Ya koi specific locality type karein)"
            ),
        },
        "options": [
            "Ramnagar Road / Moradabad Road",
            "Aliganj Road / Bazpur Road",
            "Railway Colony / Station Road",
            "Cheema Square / Awas Vikas",
        ],
    },
}


_REJECTED_NON_LOCATIONS = {
    "car", "bike", "cycle", "auto", "apple", "mango", "banana", "fruit", "food",
    "tea", "coffee", "phone", "mobile", "laptop", "computer", "tv", "skip", "next",
    "hi", "hello", "hey", "yes", "no", "ok", "okay", "house", "makan", "ghar",
    "villa", "flat", "plot", "land", "apartment", "bhk", "1bhk", "2bhk", "3bhk",
    "4bhk", "price", "budget", "cost", "money", "loan", "emi", "cash", "finance",
    "test", "testing", "asdf", "asdfg", "asdfgh", "qwerty", "zxcv", "xyz", "abc",
    "none", "nothing", "na", "n/a", "dont know", "don't know", "any", "anything", "whatever"
}

_KNOWN_GAZETTEER = {
    # Uttarakhand Cities & Towns
    "haldwani": "Haldwani",
    "kathgodam": "Kathgodam",
    "nainital": "Nainital",
    "bhimtal": "Bhimtal",
    "bhowali": "Bhowali",
    "mukteshwar": "Mukteshwar",
    "almora": "Almora",
    "ranikhet": "Ranikhet",
    "ramnagar": "Ramnagar",
    "corbett": "Jim Corbett",
    "jim corbett": "Jim Corbett",
    "dehradun": "Dehradun",
    "mussoorie": "Mussoorie",
    "rishikesh": "Rishikesh",
    "haridwar": "Haridwar",
    "roorkee": "Roorkee",
    "rudrapur": "Rudrapur",
    "kashipur": "Kashipur",
    "pantnagar": "Pantnagar",
    "kichha": "Kichha",
    "sitarganj": "Sitarganj",
    "khatima": "Khatima",
    "bazpur": "Bazpur",
    "lalkuan": "Lalkuan",
    "tanakpur": "Tanakpur",
    "champawat": "Champawat",
    "pithoragarh": "Pithoragarh",
    "bageshwar": "Bageshwar",
    "kausani": "Kausani",
    "dwarahat": "Dwarahat",
    "lansdowne": "Lansdowne",
    "kotdwar": "Kotdwar",
    "srinagar": "Srinagar (Garhwal)",
    "pauri": "Pauri",
    "tehri": "Tehri",
    "new tehri": "New Tehri",
    "uttarkashi": "Uttarkashi",
    "joshimath": "Joshimath",
    "chamoli": "Chamoli",
    "rudraprayag": "Rudraprayag",
    "vikasnagar": "Vikasnagar",
    # Haldwani Localities
    "rampur road": "Rampur Road, Haldwani",
    "kaladhungi road": "Kaladhungi Road, Haldwani",
    "kaladhungi": "Kaladhungi",
    "bithoria": "Bithoria, Haldwani",
    "bithoria no 1": "Bithoria, Haldwani",
    "bithoria no 2": "Bithoria, Haldwani",
    "mukhani": "Mukhani, Haldwani",
    "choti mukhani": "Mukhani, Haldwani",
    "badi mukhani": "Mukhani, Haldwani",
    "kusumkhera": "Kusumkhera, Haldwani",
    "bareilly road": "Bareilly Road, Haldwani",
    "gas godam": "Gas Godam Road, Haldwani",
    "gas godam road": "Gas Godam Road, Haldwani",
    "pili kothi": "Pili Kothi, Haldwani",
    "kamalwaganja": "Kamalwaganja, Haldwani",
    "lalpur": "Lalpur, Haldwani",
    "lalpur nayak": "Lalpur Nayak, Haldwani",
    "unchapul": "Unchapul, Haldwani",
    "heera nagar": "Heera Nagar, Haldwani",
    "nawabi road": "Nawabi Road, Haldwani",
    "subhash nagar": "Subhash Nagar, Haldwani",
    "transport nagar": "Transport Nagar, Haldwani",
    "tikonia": "Tikonia, Haldwani",
    "judges farm": "Judges Farm, Haldwani",
    "rajpura": "Rajpura, Haldwani",
    "gaujajali": "Gaujajali, Haldwani",
    "lamachaur": "Lamachaur, Haldwani",
    "haripur nayak": "Haripur Nayak, Haldwani",
    "haripur": "Haripur, Haldwani",
    "mandi": "Mandi area, Haldwani",
    "teenpani": "Teenpani, Haldwani",
    "dahariya": "Dahariya, Haldwani",
    "fatehpur": "Fatehpur, Haldwani",
    "chorgallia": "Chorgallia",
    "damua dhunga": "Damua Dhunga, Haldwani",
    # Nainital / Bhimtal Localities
    "sattal": "Sattal",
    "sat tal": "Sattal",
    "naukuchiatal": "Naukuchiatal",
    "naukuchia tal": "Naukuchiatal",
    "mehragaon": "Mehragaon, Bhimtal",
    "bhimtal lake": "Bhimtal Lake area",
    "lake area": "Lake area, Bhimtal",
    "mallital": "Mallital, Nainital",
    "tallital": "Tallital, Nainital",
    "pangot": "Pangot, Nainital",
    "kilbury": "Kilbury, Nainital",
    "bhowali road": "Bhowali Road",
    "shyamkhet": "Shyamkhet, Bhowali",
    "ghorakhal": "Ghorakhal",
    "ramgarh": "Ramgarh",
    "jungaliya gaon": "Jungaliya Gaon, Bhimtal",
    "sukhatal": "Sukhatal, Nainital",
    "ayarpatta": "Ayarpatta, Nainital",
    # Almora Localities
    "kasar devi": "Kasar Devi, Almora",
    "bright end corner": "Bright End Corner, Almora",
    "dharanaula": "Dharanaula, Almora",
    "ranikhet road": "Ranikhet Road, Almora",
    "mall road": "Mall Road",
    "hawalbagh": "Hawalbagh, Almora",
    "pandeykhola": "Pandeykhola, Almora",
    "khatyari": "Khatyari, Almora",
    "ntd": "NTD, Almora",
    "chaubatia": "Chaubatia, Ranikhet",
    "majkhali": "Majkhali, Ranikhet",
    # Major Metros & North Indian Cities
    "delhi": "Delhi",
    "new delhi": "New Delhi",
    "noida": "Noida",
    "greater noida": "Greater Noida",
    "gurgaon": "Gurgaon",
    "gurugram": "Gurugram",
    "ghaziabad": "Ghaziabad",
    "faridabad": "Faridabad",
    "meerut": "Meerut",
    "bareilly": "Bareilly",
    "moradabad": "Moradabad",
    "rampur": "Rampur",
    "lucknow": "Lucknow",
    "kanpur": "Kanpur",
    "agra": "Agra",
    "varanasi": "Varanasi",
    "prayagraj": "Prayagraj",
    "chandigarh": "Chandigarh",
    "mohali": "Mohali",
    "panchkula": "Panchkula",
    "jaipur": "Jaipur",
    "mumbai": "Mumbai",
    "navi mumbai": "Navi Mumbai",
    "pune": "Pune",
    "bangalore": "Bangalore",
    "bengaluru": "Bengaluru",
    "hyderabad": "Hyderabad",
    "chennai": "Chennai",
    "kolkata": "Kolkata",
    "ahmedabad": "Ahmedabad",
    "indore": "Indore",
    "bhopal": "Bhopal",
}


def verify_location_with_gemini(text: str) -> tuple[bool, str]:
    """Verify if user input is a recognized geographical place in India using Gemini AI."""
    gemini_key, gemini_model = _get_gemini_config()
    if not gemini_key:
        return False, text

    prompt = (
        f"You are a geographical verification engine for India (specifically Uttarakhand, NCR, and Indian cities/towns/villages).\n"
        f"Determine if '{text}' is a real, recognized city, town, village, or residential colony/locality in India.\n"
        f"Respond ONLY in valid JSON format:\n"
        f"{{\n"
        f'  "is_valid": true or false,\n'
        f'  "place_name": "<Canonical, correctly capitalized place name if valid, or empty string if false>"\n'
        f"}}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1,
            "maxOutputTokens": 100,
        },
    }
    models = _get_candidate_models(gemini_model)
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
            resp = _ai_session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=(2.5, 4.5))
            if resp.status_code == 429:
                _mark_model_exhausted(model, 600.0)
                continue
            if not resp.ok:
                continue
            body = resp.json()
            raw = body.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            parsed = _extract_json_object(raw)
            if parsed and "is_valid" in parsed:
                is_valid = bool(parsed["is_valid"])
                canonical = str(parsed.get("place_name") or text.title()).strip()
                return is_valid, canonical if is_valid else text
        except Exception:
            continue
    return False, text


def is_valid_location(user_text: str) -> tuple[bool, str]:
    """Verify whether user_text is a valid recognized city, town, or locality.
    Returns (is_valid: bool, canonical_name: str).
    """
    if not user_text:
        return False, ""
    clean = user_text.strip()
    clean_low = clean.lower()

    # 1. Sanity & non-location rejection filter
    if len(clean) < 2 or len(clean) > 80:
        return False, clean
    if re.fullmatch(r"[\d\s\-\+\(\)\,\.\#\/]+", clean):
        return False, clean
    if clean_low in _REJECTED_NON_LOCATIONS:
        return False, clean
    # Check if string contains only gibberish consonants (e.g. 'asdfgh', 'qwrty')
    if len(clean) >= 5 and not re.search(r"[aeiouy\u0900-\u097F]", clean_low):
        return False, clean

    # 2. Check in-memory gazetteer (fast exact or substring match)
    if clean_low in _KNOWN_GAZETTEER:
        return True, _KNOWN_GAZETTEER[clean_low]
    for k, v in _KNOWN_GAZETTEER.items():
        if len(k) >= 4 and (k == clean_low or k in clean_low or clean_low in k):
            return True, v

    # 3. Check with Google Gemini AI
    is_valid_ai, place_name = verify_location_with_gemini(clean)
    if is_valid_ai and place_name:
        return True, place_name

    # 4. Fallback to location_service (Photon / OSM) if available
    try:
        from location_service import verified_locations
        locs = verified_locations(clean)
        if locs:
            return True, clean.title()
    except Exception:
        pass

    return False, clean


def _fetch_gemini_locality_suggestions(place_name: str) -> dict | None:
    """Use Gemini AI to dynamically generate 4 prime localities and a warm welcoming intro for any city/location."""
    key, model = _get_gemini_config()
    if not key:
        return None

    clean = place_name.strip()
    prompt = (
        f"You are an expert real estate consultant in India.\n"
        f"The customer wants to buy property or a home in '{clean}'.\n"
        f"Provide exactly 4 prime residential or investment localities/areas in or around '{clean}'.\n"
        f"Also provide a warm, 1-sentence welcoming intro praising this location.\n"
        f"Return ONLY a valid JSON object matching this schema:\n"
        f"{{\n"
        f'  "title": "{clean.title()}",\n'
        f'  "intro": "<Warm 1-sentence welcoming intro in English>",\n'
        f'  "intro_hi": "<Same intro in Hindi>",\n'
        f'  "intro_hinglish": "<Same intro in Hinglish>",\n'
        f'  "options": [\n'
        f'     "<Locality 1>",\n'
        f'     "<Locality 2>",\n'
        f'     "<Locality 3>",\n'
        f'     "<Locality 4>"\n'
        f"  ]\n"
        f"}}"
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 450,
        },
    }
    models = [model, "gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest"]
    for m in models:
        if not m:
            continue
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
            resp = _ai_session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=(3.0, 8.0))
            if not resp.ok:
                continue
            raw = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            data = _extract_json_object(raw)
            if data and data.get("options") and len(data["options"]) >= 3:
                title = data.get("title") or clean.title()
                intro_en = data.get("intro") or f"{title} offers great residential and investment opportunities! 🌟"
                intro_hi = data.get("intro_hi") or f"{title} रहने और प्रॉपर्टी निवेश के लिए एक बहुत ही शानदार विकल्प है! 🌟"
                intro_hinglish = data.get("intro_hinglish") or f"{title} ek bohot badiya jagah hai with great living and investment potential! 🌟"
                options = [str(o).strip() for o in data["options"][:4] if str(o).strip()]

                q_en = (
                    f"Which specific area in {title} do you prefer?\n"
                    + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
                    + "\n(Or reply with any specific colony/locality you prefer)"
                )
                q_hi = (
                    f"{title} में आप किस खास इलाके में प्रॉपर्टी देख रहे हैं?\n"
                    + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
                    + "\n(अथवा अपनी पसंद का इलाका लिखें)"
                )
                q_hinglish = (
                    f"{title} me aap kis specific area ko prefer karte hain?\n"
                    + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
                    + "\n(Ya koi specific locality type karein)"
                )

                return {
                    "title": title,
                    "intro": {
                        "en": intro_en,
                        "hi": intro_hi,
                        "hinglish": intro_hinglish,
                    },
                    "question": {
                        "en": q_en,
                        "hi": q_hi,
                        "hinglish": q_hinglish,
                    },
                    "options": options,
                }
        except Exception:
            continue
    return None


def get_locality_data_or_ai_suggestions(place_name: str, lang: str = "en") -> dict:
    """Retrieve at least 3-4 prime locality options for the given city or location.
    Checks curated LOCALITY_SUGGESTIONS first; if not found, queries Gemini AI dynamically.
    Guarantees at least 3-4 options in the returned dictionary.
    """
    clean = (place_name or "").strip()
    clean_low = clean.lower()

    # 1. Check exact or substring match in curated LOCALITY_SUGGESTIONS
    if clean_low in LOCALITY_SUGGESTIONS:
        return LOCALITY_SUGGESTIONS[clean_low]
    for k, data in LOCALITY_SUGGESTIONS.items():
        if len(k) >= 4 and (k == clean_low or k in clean_low or clean_low in k):
            return data

    # 2. Check known aliases (e.g., Corbett -> Ramnagar)
    _ALIASES = {
        "corbett": "ramnagar",
        "jim corbett": "ramnagar",
        "kathgodam": "haldwani",
        "doon": "dehradun",
        "naini": "nainital",
    }
    for alias, target in _ALIASES.items():
        if alias in clean_low and target in LOCALITY_SUGGESTIONS:
            return LOCALITY_SUGGESTIONS[target]

    # 3. Dynamic Gemini AI suggestions for ANY custom city / town
    ai_data = _fetch_gemini_locality_suggestions(clean)
    if ai_data and len(ai_data.get("options", [])) >= 3:
        return ai_data

    # 4. Intelligent Fallback Template if AI is offline
    title = clean.title() or "Selected Area"
    opts = [
        f"{title} Central / Mall Road Area",
        f"{title} Highway / Expressway Corridor",
        f"{title} Prime Residential Sector / Civil Lines",
        f"{title} Greenbelt / Suburban Surroundings",
    ]
    return {
        "title": title,
        "intro": {
            "en": f"{title} is a wonderful place with great residential and investment opportunities! 🌟",
            "hi": f"{title} रहने और प्रॉपर्टी निवेश के लिए एक बहुत ही शानदार विकल्प है! 🌟",
            "hinglish": f"{title} ek bohot badhiya jagah hai with great residential and investment opportunities! 🌟",
        },
        "question": {
            "en": (
                f"Which specific area in {title} do you prefer?\n"
                f"1. {opts[0]}\n"
                f"2. {opts[1]}\n"
                f"3. {opts[2]}\n"
                f"4. {opts[3]}\n"
                f"(Or reply with any specific colony/locality you prefer)"
            ),
            "hi": (
                f"{title} में आप किस खास इलाके में प्रॉपर्टी देख रहे हैं?\n"
                f"1. {opts[0]}\n"
                f"2. {opts[1]}\n"
                f"3. {opts[2]}\n"
                f"4. {opts[3]}\n"
                f"(अथवा अपनी पसंद का इलाका लिखें)"
            ),
            "hinglish": (
                f"{title} me aap kis specific area ko prefer karte hain?\n"
                f"1. {opts[0]}\n"
                f"2. {opts[1]}\n"
                f"3. {opts[2]}\n"
                f"4. {opts[3]}\n"
                f"(Ya koi specific locality type karein)"
            ),
        },
        "options": opts,
    }


def resolve_locality_choice(user_msg: str, city_key: str, dynamic_options: list | None = None) -> tuple[bool, str]:
    """Resolve user selection (1-4 or locality name) for specific area within chosen city.
    Returns (is_valid: bool, resolved_locality: str).
    """
    msg = user_msg.strip()
    options = []
    if dynamic_options and isinstance(dynamic_options, list):
        options = dynamic_options
    elif city_key:
        data = LOCALITY_SUGGESTIONS.get(city_key)
        if not data:
            for k, v in LOCALITY_SUGGESTIONS.items():
                if k in city_key or city_key in k:
                    data = v
                    break
        if data:
            options = data.get("options", [])

    if not options:
        return is_valid_location(msg)

    # 1. Match numeric option (1-4 or 1-opts_len, including Hindi numerals)
    m = re.match(r"^(?:option\s*|#\s*|no\.?\s*)?([1-9]|1[0-5]|१[०-५]|[१-९])[\.\)]?$", msg, re.IGNORECASE)
    if m:
        raw_digit = m.group(1)
        num_val = _HINDI_NUMS.get(raw_digit)
        if num_val is None:
            try:
                num_val = int(raw_digit)
            except ValueError:
                num_val = None
        if num_val is not None:
            idx = num_val - 1
            if 0 <= idx < len(options):
                return True, options[idx]

    # 2. Match text substring against option names
    msg_low = msg.lower()
    for opt in options:
        opt_low = opt.lower()
        parts = [p.strip() for p in re.split(r"[/,()–—]", opt_low)]
        if any(p in msg_low for p in parts if len(p) >= 4) or opt_low in msg_low:
            return True, opt

    # 3. Match free-form custom colony / area entered by user
    valid, loc_name = is_valid_location(msg)
    if valid:
        return True, loc_name

    return False, msg


def _generate_assistant_acknowledgement(slot: str, value: str, lang: str = "en") -> str:
    """Generate intelligent consultative AI assistant acknowledgement for customer selections."""
    val_l = str(value or "").lower()
    if slot == "location_preference":
        if "rampur" in val_l:
            return {
                "hi": "रामपुर रोड हल्द्वानी का प्रमुख और सबसे तेजी से विकसित होता क्षेत्र है! 🚗",
                "hinglish": "Rampur Road Haldwani ka prime corridor hai with excellent road connectivity aur commercial access! 🚗",
                "en": "Rampur Road is one of Haldwani's fastest growing corridors with superb highway connectivity and daily conveniences! 🚗",
            }.get(lang, "Rampur Road is one of Haldwani's fastest growing corridors with superb highway connectivity! 🚗")
        if "kaladhungi" in val_l:
            return {
                "hi": "कालाढूंगी रोड शांत आवासीय वातावरण, हरियाली और अच्छे स्कूलों के लिए जाना जाता है! 🌿",
                "hinglish": "Kaladhungi Road peaceful residential surroundings, greenery aur top schools ke liye famous hai! 🌿",
                "en": "Kaladhungi Road is renowned for its peaceful residential vibe, green surroundings, and top schools! 🌿",
            }.get(lang, "Kaladhungi Road is renowned for its peaceful residential vibe and green surroundings! 🌿")
        if any(w in val_l for w in ["bithoria", "mukhani"]):
            return {
                "hi": "बिठौरिया और मुखानी हल्द्वानी के सबसे लोकप्रिय और प्रीमियम आवासीय इलाके हैं! 🏡",
                "hinglish": "Bithoria aur Mukhani Haldwani ke premier family residential hubs hain! 🏡",
                "en": "Bithoria & Mukhani are premier residential hubs—very popular with families for their convenience and amenities! 🏡",
            }.get(lang, "Bithoria & Mukhani are premier residential hubs for families! 🏡")
        if any(w in val_l for w in ["kusumkhera", "kathgodam"]):
            return {
                "hi": "कुसुमखेड़ा और काठगोदाम रेलवे स्टेशन और पहाड़ियों की तलहटी के समीप उत्तम विकल्प हैं! 🚂",
                "hinglish": "Kusumkhera aur Kathgodam station ke paas aur foothill weather ke liye best hain! 🚂",
                "en": "Kusumkhera & Kathgodam offer pleasant foothill weather and close proximity to the railway station! 🚂",
            }.get(lang, "Kusumkhera & Kathgodam offer pleasant foothill weather! 🚂")
        if any(w in val_l for w in ["nainital", "bhimtal", "bhowali"]):
            return {
                "hi": "नैनीताल और भीमताल साल भर सुहावने मौसम और खूबसूरत हिमालयी दृश्यों के लिए प्रसिद्ध हैं! 🌄",
                "hinglish": "Nainital aur Bhimtal scenic mountain views aur peaceful climate ke liye perfect hain! 🌄",
                "en": "Nainital & Bhimtal offer refreshing weather and scenic Himalayan vistas all year round! 🌄",
            }.get(lang, "Nainital & Bhimtal offer refreshing weather and scenic vistas! 🌄")
        if "almora" in val_l:
            return {
                "hi": "अल्मोड़ा शांत पहाड़ी जीवन और समृद्ध सांस्कृतिक विरासत का केंद्र है! 🏔️",
                "hinglish": "Almora serene mountain retreat aur panoramic Himalayan views ke liye famous hai! 🏔️",
                "en": "Almora offers a serene retreat lifestyle with peaceful mountain views and rich culture! 🏔️",
            }.get(lang, "Almora offers a serene retreat lifestyle with peaceful mountain views! 🏔️")
        if any(w in val_l for w in ["dehradun", "rajpur", "sahastradhara"]):
            return {
                "hi": "देहरादून की दून घाटी, प्रतिष्ठित स्कूल और उत्कृष्ट कनेक्टिविटी इसे रहने के लिए सर्वोत्तम बनाते हैं! 🌟",
                "hinglish": "Dehradun scenic Doon valley, premier schools aur high-growth connectivity ke liye perfect hai! 🌟",
                "en": "Dehradun offers premier schools, scenic Doon valley living, and rapid modern connectivity! 🌟",
            }.get(lang, "Dehradun offers premier schools and scenic Doon valley living! 🌟")
        if any(w in val_l for w in ["rishikesh", "tapovan", "ganga"]):
            return {
                "hi": "ऋषिकेश में मां गंगा के सानिध्य में शांतिपूर्ण और आध्यात्मिक जीवन का अनूठा अनुभव मिलता है! 🧘",
                "hinglish": "Rishikesh Holy Ganga ke bank pe spiritual peace aur lucrative rentals ke liye unmatched hai! 🧘",
                "en": "Rishikesh provides spiritual serenity and high-yield holiday rental potential by the Holy Ganga! 🧘",
            }.get(lang, "Rishikesh provides spiritual serenity by the Holy Ganga! 🧘")
        if any(w in val_l for w in ["haridwar", "shivalik", "bhel"]):
            return {
                "hi": "हरिद्वार में धार्मिक आस्था के साथ-साथ उत्कृष्ट हाईवे कनेक्टिविटी और आधुनिक सुविधाएं उपलब्ध हैं! 🕉️",
                "hinglish": "Haridwar sacred heritage ke saath top highway infrastructure aur planned township living offer karta hai! 🕉️",
                "en": "Haridwar combines sacred heritage with premier highway infrastructure and planned living! 🕉️",
            }.get(lang, "Haridwar combines sacred heritage with premier highway infrastructure! 🕉️")
        if any(w in val_l for w in ["mussoorie", "landour"]):
            return {
                "hi": "मसूरी की ठंडी वादियों और पहाड़ों की रानी का जीवन सचमुच एक स्वप्निल अनुभव है! 🏔️",
                "hinglish": "Mussoorie scenic valleys aur Queen of Hills lifestyle dream holiday-home living deta hai! 🏔️",
                "en": "Mussoorie delivers breathtaking valley views and prestigious Queen of Hills living! 🏔️",
            }.get(lang, "Mussoorie delivers breathtaking valley views and prestigious living! 🏔️")
        if any(w in val_l for w in ["mukteshwar", "chauli ki jali"]):
            return {
                "hi": "मुक्तेश्वर में हिमालय की बर्फीली चोटियों के 180-डिग्री दृश्य और सेब के बाग मन मोह लेते हैं! 🍎",
                "hinglish": "Mukteshwar 180-degree Himalayan snow views aur apple orchards ke liye unmatched hai! 🍎",
                "en": "Mukteshwar offers unmatched 180-degree Himalayan snow-peak panoramas and serene apple orchards! 🍎",
            }.get(lang, "Mukteshwar offers unmatched Himalayan snow panoramas! 🍎")
        if any(w in val_l for w in ["ranikhet", "majkhali"]):
            return {
                "hi": "रानीखेत का गोल्फ कोर्स और देवदार के घने जंगल शांत पहाड़ी जीवन के लिए बेमिसाल हैं! ⛳",
                "hinglish": "Ranikhet lush golf grounds aur peaceful pine forests ke saath serene mountain living deta hai! ⛳",
                "en": "Ranikhet is renowned for its lush golf course and tranquil pine forest surroundings! ⛳",
            }.get(lang, "Ranikhet is renowned for its lush golf course and pine forest serenity! ⛳")
        if any(w in val_l for w in ["ramnagar", "corbett"]):
            return {
                "hi": "रामनगर और जिम कॉर्बेट का हरा-भरा वातावरण और वाइल्डलाइफ रिसॉर्ट्स अद्भुत अनुभव कराते हैं! 🐯",
                "hinglish": "Ramnagar aur Corbett riverfront eco-resorts aur green forest living ke liye top choice hai! 🐯",
                "en": "Ramnagar and Jim Corbett offer prestigious riverfront living and eco-resort opportunities! 🐯",
            }.get(lang, "Ramnagar and Jim Corbett offer prestigious riverfront living! 🐯")
        if any(w in val_l for w in ["rudrapur", "kashipur"]):
            return {
                "hi": "रुद्रपुर और काशीपुर उत्तराखंड के प्रमुख औद्योगिक व व्यापारिक केंद्र हैं, जिनमें असीम संभावनाएं हैं! 🏭",
                "hinglish": "Rudrapur aur Kashipur booming commercial hubs aur strong rental demand ke liye best hain! 🏭",
                "en": "Rudrapur & Kashipur are booming industrial hubs offering high rental yields and modern townships! 🏭",
            }.get(lang, "Rudrapur & Kashipur offer high growth and modern townships! 🏭")
        return {
            "hi": f"{value} एक बहुत ही बेहतरीन विकल्प है! 🌟",
            "hinglish": f"{value} ek bohot achha choice hai! 🌟",
            "en": f"Great choice of location in {value}! 🌟",
        }.get(lang, f"Great choice of location in {value}! 🌟")

    elif slot == "planning_to_buy":
        if "immediately" in val_l:
            return {
                "hi": "तुरंत ख़रीदने की योजना बहुत अच्छी है—हम रेडी-टू-मूव प्रॉपर्टीज़ को प्राथमिकता देंगे! ⏱️",
                "hinglish": "Immediately plan karna bohot badiya hai—hum ready-to-move options prioritize karenge! ⏱️",
                "en": "Buying immediately is a great priority—we will focus on ready-to-move properties with immediate possession! ⏱️",
            }.get(lang, "Buying immediately is a great priority! ⏱️")
        if "1 month" in val_l:
            return {
                "hi": "1 महीने की समयसीमा उत्तम है—हम आपके लिए चुनिंदा प्रॉपर्टीज़ तैयार रखेंगे! 📅",
                "hinglish": "Within 1 month ka timeline perfect hai shortlist karke visits arrange karne ke liye! 📅",
                "en": "A timeline of within 1 month gives us the perfect window to shortlist and visit prime units for you! 📅",
            }.get(lang, "A timeline of within 1 month gives us the perfect window to shortlist prime units! 📅")
        if "3 months" in val_l:
            return {
                "hi": "3 महीने की योजना से हम तैयार और निर्माणाधीन दोनों बेहतरीन प्रोजेक्ट देख सकते हैं! 👍",
                "hinglish": "Within 3 months ka time humein ready aur near-completion dono options explore karne ka mauka deta hai! 👍",
                "en": "Planning within 3 months allows us to explore both ready homes and near-possession projects! 👍",
            }.get(lang, "Planning within 3 months allows us to explore prime options! 👍")
        if any(w in val_l for w in ["6 months", "12 months"]):
            return {
                "hi": "6 से 12 महीने की समयसीमा से आपको शुरुआती दरों पर बेहतरीन प्रोजेक्ट बुक करने का लाभ मिलता है! 📈",
                "hinglish": "6 se 12 months ka timeline early-stage pricing aur attractive payment plans ke liye best hai! 📈",
                "en": "A timeline within 6 to 12 months offers great flexibility to secure early-stage pricing! 📈",
            }.get(lang, "A timeline within 6 to 12 months offers great flexibility! 📈")
        return {
            "hi": "समय रहते रिसर्च करना समझदारी भरा कदम है—इससे सही निर्णय लेने में मदद मिलती है! 🔍",
            "hinglish": "Pehle se explore karna bohot smart hai market trends samajhne ke liye! 🔍",
            "en": "Exploring early is smart—it helps you understand market trends and find the right investment! 🔍",
        }.get(lang, "Exploring early is smart to understand market trends! 🔍")

    elif slot == "budget_range":
        if "above 2 crore" in val_l:
            return {
                "hi": "2 करोड़ से अधिक का बजट आपको अल्ट्रा-लक्ज़री और भव्य विला सेगमेंट में स्थान देता है! ✨",
                "hinglish": "Above 2 crore aapko ultra-luxury hill estates aur premium villas me position karta hai! ✨",
                "en": "Above 2 crore puts you in the ultra-luxury segment with expansive hillside estates and bespoke villas! ✨",
            }.get(lang, "Above 2 crore puts you in the ultra-luxury segment! ✨")
        if any(w in val_l for w in ["1 crore - 2 crore", "1 crore to 2 crore", "1 - 2 crore"]):
            return {
                "hi": "1 से 2 करोड़ का बजट प्रीमियम प्रॉपर्टीज़, लक्ज़री विला और बड़े प्लॉट्स के द्वार खोलता है! 💎",
                "hinglish": "1 se 2 crore ka budget premium villas, independent houses aur prime plots ke liye best hai! 💎",
                "en": "A budget of 1 crore - 2 crore gives you access to premium properties, luxury villas, and prime plots! 💎",
            }.get(lang, "A budget of 1 crore - 2 crore gives you access to premium properties! 💎")
        if "50 lakh - 1 crore" in val_l:
            return {
                "hi": "50 लाख से 1 करोड़ का बजट परिवारों में बहुत लोकप्रिय है—इसमें बेहतरीन 2 व 3 BHK फ्लैट्स उपलब्ध हैं! 🏡",
                "hinglish": "50 lakh se 1 crore bohot popular range hai with top 2 aur 3 BHK gated society options! 🏡",
                "en": "50 lakh to 1 crore is a very popular range with excellent 2 & 3 BHK gated society flats and independent houses! 🏡",
            }.get(lang, "50 lakh to 1 crore is a very popular range with excellent options! 🏡")
        if "25 lakh - 50 lakh" in val_l:
            return {
                "hi": "25 से 50 लाख का बजट 1 व 2 BHK अपार्टमेंट और आवासीय प्लॉट्स के लिए बहुत अच्छा है! 🏠",
                "hinglish": "25 se 50 lakh me bohot acche 1 & 2 BHK flats aur residential plots available hain! 🏠",
                "en": "A budget of 25 to 50 lakh opens up very good 1 BHK and 2 BHK apartments and residential plots! 🏠",
            }.get(lang, "A budget of 25 to 50 lakh opens up very good options! 🏠")
        return {
            "hi": "25 लाख तक के बजट में किफायती प्लॉट्स और अपार्टमेंट के अच्छे विकल्प मौजूद हैं! 💰",
            "hinglish": "Up to 25 lakh me developing areas me budget-friendly plots aur flats available hain! 💰",
            "en": "A budget up to 25 lakh has solid options in developing sectors and affordable plots/apartments! 💰",
        }.get(lang, "A budget up to 25 lakh has solid options! 💰")

    elif slot == "payment_option":
        if "cash" in val_l:
            return {
                "hi": "पूरा नकद भुगतान त्वरित रजिस्ट्री और आकर्षक छूट सुनिश्चित करता है! 💵",
                "hinglish": "Full Cash Payment smooth process, quick registration aur maximum discounts ensure karta hai! 💵",
                "en": "Full Cash Payment ensures smooth, expedited transactions and great developer negotiations! 💵",
            }.get(lang, "Full Cash Payment ensures smooth, expedited transactions! 💵")
        if any(w in val_l for w in ["loan", "emi", "finance"]):
            return {
                "hi": "बैंक फाइनेंस / होम लोन आसान ईएमआई के साथ एक सुरक्षित और सुविधाजनक विकल्प है! 🏦",
                "hinglish": "Bank Finance / Home loan EMI ke saath budget-friendly aur comfortable option hai! 🏦",
                "en": "Bank Finance / Home Loan is a great, manageable route with convenient EMI options! 🏦",
            }.get(lang, "Bank Finance / Home Loan is a great route with convenient EMI options! 🏦")
        if "nri" in val_l:
            return {
                "hi": "हमारा एनआरआई डेस्क पूरी लीगल जांच और बैंकिंग सहायता प्रदान करता है! 🌐",
                "hinglish": "Hamara dedicated NRI desk complete legal verification aur banking assistance deta hai! 🌐",
                "en": "Our dedicated NRI desk provides seamless end-to-end documentation and banking support! 🌐",
            }.get(lang, "Our dedicated NRI desk provides seamless support! 🌐")
        return {
            "hi": f"{value} एक बहुत ही व्यावहारिक और सुविधाजनक भुगतान योजना है! 🤝",
            "hinglish": f"{value} ek bohot convenient aur structured payment option hai! 🤝",
            "en": f"{value} is a very convenient and secure payment method! 🤝",
        }.get(lang, f"{value} is a very convenient payment method! 🤝")

    elif slot == "bhk_preference":
        if "penthouse" in val_l:
            return {
                "hi": "पेंटहाउस विशेष छत, निजी टैरेस और खूबसूरत दृश्यों का बेजोड़ अनुभव कराता है! 🌄",
                "hinglish": "Penthouse exclusive top-floor living aur panoramic mountain views deta hai! 🌄",
                "en": "A Penthouse delivers exclusive top-floor living, private terraces, and panoramic mountain vistas! 🌄",
            }.get(lang, "A Penthouse delivers exclusive top-floor living and panoramic vistas! 🌄")
        if "villa" in val_l:
            return {
                "hi": "विला में आपको निजी गार्डन, पार्किंग और खुली जगह का आनंद मिलता है! 🏡",
                "hinglish": "Villa me complete privacy, personal garden aur dedicated parking milti hai! 🏡",
                "en": "A Villa offers private open spaces, dedicated parking, and peaceful foothill living! 🏡",
            }.get(lang, "A Villa offers private open spaces and dedicated parking! 🏡")
        if re.search(r"3\.5\s*bhk", val_l):
            return {
                "hi": "3.5 BHK एक बेहतरीन विकल्प है - 3 बड़े कमरों के साथ स्टडी या अतिरिक्त उपयोग के लिए उत्तम! 🏡",
                "hinglish": "3.5 BHK bohot badhiya choice hai - 3 bedrooms ke saath dedicated study ya extra room milta hai! 🏡",
                "en": "3.5 BHK is a wonderful choice - giving you 3 spacious bedrooms plus a dedicated study or multipurpose room! 🏡",
            }.get(lang, "3.5 BHK is a wonderful choice with a dedicated study room! 🏡")
        if re.search(r"2\.5\s*bhk", val_l):
            return {
                "hi": "2.5 BHK एक स्मार्ट विकल्प है - 2 कमरों के साथ वर्क-फ्रॉम-होम या स्टडी रूम की अतिरिक्त सुविधा! 🛋️",
                "hinglish": "2.5 BHK ek smart choice hai - 2 bedrooms ke saath personal study ya workspace milta hai! 🛋️",
                "en": "2.5 BHK is a smart choice - offering 2 comfortable bedrooms plus a flexible study or work-from-home space! 🛋️",
            }.get(lang, "2.5 BHK is a smart choice with a flexible study space! 🛋️")
        if any(w in val_l for w in ["5 bhk+", "5+ bhk"]) or re.search(r"(?<![\d\.])5\s*bhk", val_l):
            return {
                "hi": "5 BHK+ भव्य, विशाल और अल्ट्रा-लक्ज़री जीवनशैली के लिए एक शानदार विकल्प है! 🏰",
                "hinglish": "5 BHK+ expansive space, grand living aur luxury lifestyle ke liye unmatched choice hai! 🏰",
                "en": "5 BHK+ is a premier choice - providing grand, ultra-luxurious living with expansive space for your family! 🏰",
            }.get(lang, "5 BHK+ is a premier choice providing grand, luxurious living! 🏰")
        if re.search(r"(?<![\d\.])4\s*bhk", val_l):
            return {
                "hi": "4 BHK एक शानदार विकल्प है - विशाल, लक्ज़री और पूरे परिवार के लिए भरपूर स्पेस और प्राइवेसी! 🏰",
                "hinglish": "4 BHK bohot accha choice hai - spacious luxury living, family comfort aur extra space ke liye perfect! 🏰",
                "en": "4 BHK is an exceptional choice - offering spacious luxury living, generous family areas, and dedicated space for work and leisure! 🏰",
            }.get(lang, "4 BHK is an exceptional choice offering spacious, luxurious living! 🏰")
        if re.search(r"(?<![\d\.])3\s*bhk", val_l):
            return {
                "hi": "3 BHK परिवार के लिए पर्याप्त जगह, आराम और प्राइवेसी प्रदान करता है! 👨‍👩‍👧‍👦",
                "hinglish": "3 BHK family ke liye generous living space, comfort aur privacy provide karta hai! 👨‍👩‍👧‍👦",
                "en": "3 BHK is a wonderful choice - providing generous living areas, extra comfort, and privacy for your family! 👨‍👩‍👧‍👦",
            }.get(lang, "3 BHK homes provide generous living areas and extra comfort! 👨‍👩‍👧‍👦")
        if re.search(r"(?<![\d\.])2\s*bhk", val_l):
            return {
                "hi": "2 BHK एक आरामदायक और व्यावहारिक विकल्प है - आधुनिक जीवनशैली और सुविधा के लिए उत्तम! 🛋️",
                "hinglish": "2 BHK comfortable living aur convenience ke liye ek fantastic choice hai! 🛋️",
                "en": "2 BHK is a practical and comfortable choice - offering modern living with great everyday convenience! 🛋️",
            }.get(lang, "2 BHK is a practical and comfortable choice for modern living! 🛋️")
        if re.search(r"1(\.5)?\s*bhk", val_l):
            return {
                "hi": "1 BHK एक किफायती और बेहतरीन विकल्प है - आसान रखरखाव और उत्तम जीवनशैली! 🏢",
                "hinglish": "1 BHK compact living aur easy maintenance ke liye bohot accha choice hai! 🏢",
                "en": "1 BHK is an excellent choice - compact, easy to maintain, and offering great living efficiency! 🏢",
            }.get(lang, "1 BHK is an excellent choice for compact, efficient living! 🏢")
        if re.search(r"1\s*rk", val_l):
            return {
                "hi": "1 RK एक किफायती और व्यावहारिक विकल्प है - सरल रखरखाव और बेहतरीन उपयोग! 🏢",
                "hinglish": "1 RK budget-friendly aur low maintenance living ke liye smart option hai! 🏢",
                "en": "1 RK is a very smart, economical choice - efficient, compact, and low maintenance! 🏢",
            }.get(lang, "1 RK is a smart, economical choice! 🏢")
        return {
            "hi": f"{value} आपकी प्रॉपर्टी खोज के लिए एक शानदार विकल्प है! 🏢",
            "hinglish": f"{value} aapki property search ke liye fantastic choice hai! 🏢",
            "en": f"{value} is a fantastic choice for comfortable living! 🏢",
        }.get(lang, f"{value} is a fantastic choice for comfortable living! 🏢")

    # Acknowledgements for feedback, ratings, and custom admin questions
    if any(k in slot.lower() for k in ["feedback", "star", "rating", "review"]):
        return {
            "hi": "आपकी बहुमूल्य प्रतिक्रिया के लिए बहुत-बहुत धन्यवाद! ⭐",
            "hinglish": "Aapke valuable feedback ke liye bohot shukriya! ⭐",
            "en": "Thank you so much for your valuable feedback! ⭐",
        }.get(lang, "Thank you so much for your valuable feedback! ⭐")

    if slot == "remarks_requirements":
        if val_l in {"none", "skip", "no", "nahi", "kuch nahi", "-", "null"}:
            return ""
        return {
            "hi": "धन्यवाद, हमने आपकी विशेष आवश्यकताएं नोट कर ली हैं। ✍️",
            "hinglish": "Thank you, humne aapki specific requirements note kar li hain! ✍️",
            "en": "Thank you, I have noted your specific requirements! ✍️",
        }.get(lang, "Thank you, I have noted your specific requirements! ✍️")

    if slot not in {"location_preference", "planning_to_buy", "budget_range", "payment_option", "bhk_preference", "remarks_requirements"}:
        return {
            "hi": "धन्यवाद, हमने आपकी यह जानकारी नोट कर ली है। ✍️",
            "hinglish": "Thank you, humne yeh details note kar li hain. ✍️",
            "en": "Thank you, I have noted that down! ✍️",
        }.get(lang, "Thank you, I have noted that down! ✍️")

    return ""


def extract_bhk_from_text(msg_lower: str) -> str | None:
    """Extract standard BHK/property configuration from text."""
    if any(w in msg_lower for w in ["1 rk", "1rk"]):
        return "1 RK"
    elif any(w in msg_lower for w in ["2.5 bhk", "2.5bhk", "2.5 बीएचके"]):
        return "2.5 BHK"
    elif any(w in msg_lower for w in ["3.5 bhk", "3.5bhk", "3.5 बीएचके"]):
        return "3.5 BHK"
    elif any(w in msg_lower for w in ["1 bhk", "1bhk", "1 बीएचके"]):
        return "1 BHK"
    elif any(w in msg_lower for w in ["2 bhk", "2bhk", "2 बीएचके"]):
        return "2 BHK"
    elif any(w in msg_lower for w in ["3 bhk", "3bhk", "3 बीएचके"]):
        return "3 BHK"
    elif any(w in msg_lower for w in ["4 bhk", "4bhk", "4 बीएचके"]):
        return "4 BHK"
    elif any(w in msg_lower for w in ["5 bhk", "5bhk", "5 bhk+", "5+ bhk", "5 बीएचके"]):
        return "5 BHK+"
    elif any(w in msg_lower for w in ["penthouse"]):
        return "Penthouse"
    elif any(w in msg_lower for w in ["villa", "kothi", "bungalow", "ghar", "makan", "मकान", "घर"]):
        return "Villa"
    return None


def _resolve_numbered_choice(user_message: str, slot: str) -> str | None:
    """Resolve a numerical reply (1-15 or Hindi numerals) to the corresponding question option."""
    txt = user_message.strip()

    # If slot is bhk_preference and user mentioned a BHK configuration (e.g. "4 BHK", "3 BHK", "Villa"),
    # do NOT resolve it as a numerical menu index.
    if slot == "bhk_preference" and re.search(r"\b(?:bhk|rk|bed|bedroom|villa|penthouse)\b", txt, re.IGNORECASE):
        return None

    match = re.match(
        r"^(?:option\s*|#\s*|no\.?\s*|number\s*)?([1-9]|1[0-5]|१[०-५]|[१-९])[\.\)]?(?:\s*(?:st|nd|rd|th|number))?$",
        txt,
        re.IGNORECASE,
    )
    if not match:
        return None
    raw_digit = match.group(1)
    num_val = _HINDI_NUMS.get(raw_digit, None)
    if num_val is None:
        try:
            num_val = int(raw_digit)
        except ValueError:
            return None
    idx = num_val - 1
    options = []
    try:
        active_qs = get_active_questions_sync()
        for q in active_qs:
            if q.get("slot_key") == slot:
                opts = q.get("options")
                if opts and isinstance(opts, list):
                    options = opts
                    break
    except Exception:
        pass
    if not options:
        options = STANDARD_OPTIONS.get(slot, [])
    if 0 <= idx < len(options):
        return options[idx]
    return None


def _format_bullet_options(options: list[str]) -> str:
    """Format options as clean, modern WhatsApp bullet points."""
    return "\n".join(f"• {opt}" for opt in options)


def _strip_repetitive_greetings(text: str) -> str:
    """Strip repetitive 'Namaste [Name] ji!', 'Hi [Name]!', 'Hello [Name]!' from assistant responses."""
    if not text:
        return text

    # 1. Pattern: 'Namaste [Name] ji!' or 'Namaste [Name]!' or 'Namaste ji!' or 'Namaste!'
    p1 = r"^(?:namaste|namaskar|hello|hi|hey|greetings|welcome)(?:\s+[\w\s]{1,40}?(?:\s+ji|\bji)?)?\s*[\!.,:\-–—]\s*"
    text = re.sub(p1, "", text, flags=re.IGNORECASE).strip()

    # 2. Pattern: 'Wonderful, Shailendra deo ji! ' -> 'Wonderful! '
    p2 = r"^(wonderful|great|perfect|excellent|fantastic|got it|sure|thank you|thanks)\s*,\s*[\w\s]{1,40}?(?:\s+ji|\bji)?\s*[\!.,:\-–—]\s*"
    text = re.sub(p2, r"\1! ", text, flags=re.IGNORECASE).strip()

    # 3. Clean any remaining standalone leading 'Namaste' or 'Namaste ji' without punctuation
    text = re.sub(r"^(?:namaste|namaskar|hello|hi|hey)\s*(?:ji)?\s+", "", text, flags=re.IGNORECASE).strip()

    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def _get_calendly_schedule_block(
    cal_url: str = "",
    answers: dict | None = None,
    phone: str = "",
    lang: str = "en",
) -> str:
    """Return a prominent localized Calendly scheduling call-to-action block with both options: Meeting & Visit."""
    answers = answers or {}
    raw_meeting_url = (cal_url or os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min")).strip()
    raw_visit_url = os.getenv("CALENDLY_VISIT_URL", "https://calendly.com/rudrasinghdeo369/30min").strip()

    ctx = dict(answers)
    if phone and not ctx.get("phone"):
        ctx["phone"] = phone
        ctx["mobile_number"] = phone

    meeting_url = build_prefilled_calendly_url(raw_meeting_url, ctx)
    visit_url = build_prefilled_calendly_url(raw_visit_url, ctx)

    if lang == "hi":
        return (
            "🗓️ *आगे बढ़ने के लिए कृपया नीचे दिए गए विकल्पों में से चुनें:*\n\n"
            "1️⃣ *परामर्श मीटिंग शेड्यूल करें (Schedule a Meeting):*\n"
            f"👉 {meeting_url}\n\n"
            "2️⃣ *प्रॉपर्टी साइट विज़िट शेड्यूल करें (Schedule a Visit):*\n"
            f"👉 {visit_url}"
        )
    elif lang == "hinglish":
        return (
            "🗓️ *Aage proceed karne ke liye please neeche diye gaye options me se choose karein:*\n\n"
            "1️⃣ *Schedule a Meeting:*\n"
            f"👉 {meeting_url}\n\n"
            "2️⃣ *Schedule a Visit:*\n"
            f"👉 {visit_url}"
        )
    else:
        return (
            "🗓️ *Please choose one of the options below to proceed:*\n\n"
            "1️⃣ *Schedule a Meeting:*\n"
            f"👉 {meeting_url}\n\n"
            "2️⃣ *Schedule a Visit:*\n"
            f"👉 {visit_url}"
        )


def _prefill_all_calendly_urls_in_text(text: str, ctx: dict) -> str:
    """Ensure any Calendly URLs in text have full prefilled parameters for name, email, and phone."""
    if not text:
        return text
    raw_meeting_url = os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min").strip()
    raw_visit_url = os.getenv("CALENDLY_VISIT_URL", "https://calendly.com/rudrasinghdeo369/30min").strip()
    meeting_prefilled = build_prefilled_calendly_url(raw_meeting_url, ctx)
    visit_prefilled = build_prefilled_calendly_url(raw_visit_url, ctx)

    # Replace bare visit URL if present without query parameters
    base_visit = raw_visit_url.rstrip("/")
    text = re.sub(re.escape(base_visit) + r"(?!\?)", visit_prefilled, text)

    # Replace bare meeting URL if present without query parameters
    base_meeting = raw_meeting_url.rstrip("/")
    text = re.sub(re.escape(base_meeting) + r"(?!\?)", meeting_prefilled, text)
    return text


def _build_system_prompt(first_name: str, known_criteria: dict, is_followup: bool = False, schedule_url: str = "", visit_url: str = "") -> str:
    """Build natural conversational qualification prompt matching the Profilent specification."""
    fn = (first_name.split()[0] if first_name else "there").strip()
    booking_url = schedule_url or os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min")
    visit_booking_url = visit_url or os.getenv("CALENDLY_VISIT_URL", "https://calendly.com/rudrasinghdeo369/30min")

    active_qs = []
    try:
        active_qs = get_active_questions_sync()
    except Exception:
        pass

    if active_qs:
        slot_lines = [f'    "{q.get("slot_key", "custom")}": "..."' for q in active_qs if q.get("slot_key")]
        slots_schema_str = ",\n".join(slot_lines)
    else:
        slots_schema_str = (
            '    "location_preference": "...",\n'
            '    "planning_to_buy": "...",\n'
            '    "budget_range": "...",\n'
            '    "payment_option": "...",\n'
            '    "bhk_preference": "...",\n'
            '    "remarks_requirements": "..."'
        )

    if is_followup:
        return (
            f"You are a friendly, consultative real estate enquiry assistant for Profilent in Uttarakhand (Haldwani, Nainital, Bhimtal, Almora). You are assisting {fn} on WhatsApp.\n\n"
            f"STATUS: The customer has already completed their property questionnaire.\n"
            f"Current Known Criteria:\n{json.dumps(known_criteria, indent=2)}\n\n"
            f"RULES:\n"
            f"- AI ADVISORY ASSISTANT: Act as a helpful real estate assistant. Give insightful property suggestions and answers tailored to customer's questions (e.g. connectivity, neighborhood pros, mountain views, rental yields).\n"
            f"- DO NOT repeat greetings on every turn (no 'Namaste [Name]', 'Hi [Name]' on subsequent turns).\n"
            f"- Keep replies short, helpful, and in the customer's language (English, Hindi, or Hinglish, based on how they write).\n"
            f"- Don't quote exact prices, live availability, or loan terms. Say that a senior sales executive will confirm specific unit availability and pricing.\n"
            f"- If customer asks to speak to a person, say that a sales executive will call them shortly.\n"
            f"- When the customer wants to schedule a meeting or site visit, or when concluding, offer both Calendly scheduling options:\n"
            f"  1. Schedule a Meeting: {booking_url}\n"
            f"  2. Schedule a Visit: {visit_booking_url}\n"
            f"- CRITICAL: DO NOT invent fake site visits, fake physical addresses, morning/afternoon/evening time slots, or ask for alternate phone numbers. All scheduling must be directed to the Calendly links.\n"
            f"- If customer sends inappropriate or offensive messages, politely steer back to real estate.\n"
            f"- Keep 'is_complete': false so the conversation remains active.\n\n"
            f"Output MUST be a valid JSON object matching this schema:\n"
            f"{{\n"
            f'  "extracted_slots": {{\n'
            f"{slots_schema_str}\n"
            f"  }},\n"
            f'  "reply_message": "...",\n'
            f'  "is_complete": false\n'
            f"}}"
        )

    if active_qs:
        q_blocks = []
        for i, q in enumerate(active_qs):
            num = i + 1
            q_title = q.get("title") or q.get("slot_key", f"Question {num}")
            req_str = "(required)" if q.get("is_required", True) else "(optional)"
            q_text = q.get("question_text", "")
            block = f"{num}. {q_title} {req_str}\n   {q_text}"
            opts = q.get("options") or []
            if opts:
                opt_strs = "\n".join(f"   {oi+1}. {opt}" for oi, opt in enumerate(opts))
                block += f"\n{opt_strs}"
            guidance = q.get("ai_guidance")
            if guidance:
                block += f"\n   *GUIDANCE*: {guidance}"
            q_blocks.append(block)
        questionnaire_text = "\n\n".join(q_blocks)
    else:
        questionnaire_text = (
            "1. Location Preference (required)\n"
            "   Ask which area/city they are interested in:\n"
            "   1. Haldwani (Rampur Rd, Kaladhungi Rd, Bithoria, Mukhani)\n"
            "   2. Nainital / Bhimtal\n"
            "   3. Almora\n"
            "   4. Other (please type the area)\n"
            "   *LOCATION ADVISORY RULE*: When the customer sends any city or location name, give a warm, insightful suggestion praising that city/location, present at least 3-4 prime localities as numbered options, and confirm their choice before advancing to Question 2.\n\n"
            "2. Planning to Buy (required)\n"
            "   When are you planning to buy?\n"
            "   1. Immediately\n   2. Within 1 Month\n   3. Within 3 Months\n   4. Within 6 Months\n   5. Within 12 Months\n   6. More Than 1 Year\n   7. Just Exploring / Researching\n\n"
            "3. Budget Range (required)\n"
            "   What is your budget range?\n"
            "   1. Up to 25 lakh\n   2. 25 lakh - 50 lakh\n   3. 50 lakh - 1 crore\n   4. 1 crore - 2 crore\n   5. Above 2 crore\n\n"
            "4. Payment Option (required)\n"
            "   How do you plan to pay?\n"
            "   1. Bank Finance with EMI\n   2. Full Cash Payment\n   3. Home Loan (Mortgage)\n   4. Down Payment Plan\n"
            "   5. Construction-Linked Payment Plan (CLP)\n   6. Possession-Linked Payment Plan (PLP)\n"
            "   7. Installment Payment Plan\n   8. Developer Financing\n   9. Rent-to-Own Scheme\n   10. Part Cash + Home Loan\n"
            "   11. Part Payment with Balance at Registration\n   12. Flexible Payment Plan\n"
            "   13. Subvention Scheme (Developer Pays Pre-EMI)\n   14. Joint Home Loan (Co-applicant Financing)\n"
            "   15. NRI Payment Plan (for Overseas Buyers)\n\n"
            "5. BHK Preference (required)\n"
            "   Which configuration / BHK do you need?\n"
            "   1. 1 RK\n   2. 1 BHK\n   3. 2 BHK\n   4. 2.5 BHK\n   5. 3 BHK\n   6. 3.5 BHK\n   7. 4 BHK\n   8. 5 BHK+\n   9. Villa\n   10. Penthouse\n\n"
            "6. Remarks/Requirements (optional)\n"
            "   Do you have any specific remarks or requirements (such as parking, floor preference, facing direction, hill view, or nearby schools)? Type whatever you need, or type \"skip\" if none."
        )

    return (
        f"You are a friendly, consultative real estate enquiry assistant for Profilent in Uttarakhand (Haldwani, Nainital, Bhimtal, Almora). You are assisting {fn} on WhatsApp.\n\n"
        f"YOUR DUAL ROLE AS ASSISTANT & QUALIFIER:\n"
        f"1. AI ADVISORY ASSISTANT: Always respond with a brief, helpful suggestion or insight tailored to the customer's message before asking the next question. Examples:\n"
        f"   - If they mention Haldwani / Nainital Rd / Rampur Rd: highlight excellent road connectivity, hospitals, schools, and vibrant residential growth.\n"
        f"   - If they mention Nainital / Bhimtal: highlight scenic mountain views, peaceful vacation living, pleasant weather, and lucrative holiday rental potential.\n"
        f"   - If they mention Almora: highlight serene Himalayan vistas, cultural richness, clean environment, and peaceful retreat lifestyle.\n"
        f"   - If they mention BHK (e.g. 2 BHK / 3 BHK / Villa): validate their choice (e.g. 2 BHK is great for rental yield; 3 BHK offers generous family living; Villas provide private space).\n"
        f"   - If they mention budget or payment: reassure them that Profilent has prime verified inventory and clear documentation matching their range.\n"
        f"   - If they mention custom remarks (parking, floor, schools): acknowledge and assure that these parameters will be prioritized.\n"
        f"2. GUIDED QUESTIONNAIRE: Guide the customer step-by-step through their requirements, asking ONE question at a time in this order. Show numbered options and let the customer reply with a number or text:\n\n"
        f"{questionnaire_text}\n\n"
        f"Rules:\n"
        f"- Always provide a brief advisory suggestion/acknowledgement before asking the next question.\n"
        f"- Ask only one question per message.\n"
        f"- Required fields must be answered before you move on. If the customer skips one, politely ask again.\n"
        f"- MULTI-SLOT & ALL-IN-ONE STATEMENTS: If the customer provides multiple preferences at once (e.g. they specify location, BHK, budget, payment mode, or amenities in one message):\n"
        f"  1. Extract EVERY mentioned preference into 'extracted_slots' simultaneously.\n"
        f"  2. Provide a thoughtful analysis and personalized property suggestions based on all the provided criteria combined.\n"
        f"  3. If any required question is still missing, ask ONLY the next missing question (never re-ask for details already provided).\n"
        f"  4. If the customer provides all required information in their message, set 'is_complete': true.\n"
        f"- Don't quote prices, availability, or loan terms. Say that a sales executive will confirm these.\n"
        f"- Keep replies concise and in the customer's language (English, Hindi, or Hinglish, based on how they write).\n"
        f"- If the customer asks to speak to a person, say that a sales executive will call them and move to the summary.\n"
        f"- DO NOT repeat greetings on every turn (no 'Namaste [Name]' or 'Hi [Name]' on subsequent turns).\n"
        f"- If the customer sends inappropriate, abusive, or nonsensical messages, politely steer back to the property enquiry.\n\n"
        f"Current Known Criteria:\n{json.dumps(known_criteria, indent=2)}\n\n"
        f"When all answers are collected (or if customer asks to speak to a person), set 'is_complete': true.\n\n"
        f"Output MUST be a valid JSON object matching this schema:\n"
        f"{{\n"
        f'  "extracted_slots": {{\n'
        f"{slots_schema_str}\n"
        f"  }},\n"
        f'  "reply_message": "...",\n'
        f'  "is_complete": true | false\n'
        f"}}"
    )


def _extract_json_object(raw: str) -> dict | None:
    """Robustly extract and parse a JSON object from LLM response text."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def call_gemini_agent(
    user_message: str,
    current_answers: dict,
    recent_messages: list,
    contact_info: dict,
    is_followup: bool = False,
    schedule_url: str = "",
    visit_url: str = "",
) -> dict | None:
    """Send conversation context to Google Gemini and return structured slots and reply."""
    api_key, model_name = _get_gemini_config()
    if not api_key:
        return None

    fn = contact_info.get("first_name") or current_answers.get("first_name") or ""
    if not fn:
        full_name = contact_info.get("full_name") or current_answers.get("full_name") or "there"
        fn = full_name.split()[0] if full_name else "there"
    known_criteria = {
        "location_preference": current_answers.get("location_preference") or "",
        "bhk_preference": current_answers.get("bhk_preference") or "",
        "budget_range": current_answers.get("budget_range") or "",
        "planning_to_buy": current_answers.get("planning_to_buy") or "",
        "payment_option": current_answers.get("payment_option") or "",
        "remarks_requirements": current_answers.get("remarks_requirements") or "",
    }
    safe_user_message = re.sub(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", "[email hidden]", user_message)
    safe_user_message = re.sub(r"(?<!\w)\+?[\d][\d\s().-]{7,}[\d]", "[phone hidden]", safe_user_message)

    system_prompt = _build_system_prompt(
        fn,
        known_criteria,
        is_followup=is_followup,
        schedule_url=schedule_url,
        visit_url=visit_url,
    )

    contents = []
    for msg in recent_messages[-6:]:
        role = "model" if msg.get("role") == "assistant" else "user"
        content = (msg.get("content") or "").strip()
        if content:
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += f"\n{content}"
            else:
                contents.append({"role": role, "parts": [{"text": content}]})

    if contents and contents[-1]["role"] == "user":
        contents[-1]["parts"][0]["text"] += f"\n{safe_user_message}"
    else:
        contents.append({"role": "user", "parts": [{"text": safe_user_message}]})

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3,
            "maxOutputTokens": 600,
        },
    }

    models_to_try = _get_candidate_models(model_name)
    if not models_to_try:
        logger.info("All Gemini models currently marked exhausted/rate-limited; using high-speed local engine.")
        return None

    for current_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={api_key}"
        try:
            response = _ai_session.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=(2.5, 4.5),
            )
            if response.status_code == 429:
                _mark_model_exhausted(current_model, 600.0)
                continue
            if not response.ok:
                logger.debug("Gemini model %s returned HTTP %s", current_model, response.status_code)
                continue
            body = response.json()
            raw_text = (
                body.get("candidates", [])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            parsed = _extract_json_object(raw_text)
            if parsed:
                return parsed
        except Exception as exc:
            logger.info("Gemini model %s call retry: %s", current_model, exc)
            continue

    return None


def call_ai_agent(
    user_message: str,
    current_answers: dict,
    recent_messages: list,
    contact_info: dict,
    is_followup: bool = False,
    schedule_url: str = "",
) -> dict | None:
    """Send conversation context to Google Gemini and return structured slots and reply."""
    gemini_key, _ = _get_gemini_config()
    if gemini_key:
        return call_gemini_agent(
            user_message,
            current_answers,
            recent_messages,
            contact_info,
            is_followup=is_followup,
            schedule_url=schedule_url,
        )
    return None


_MONEY_PATTERN = re.compile(
    r"(\b\d+(\.\d+)?\s*(lakh|lac|cr|crore|k|thousand|rupee|rs|inr|₹)\b|\/mo|\bper month\b|\brent\b|\bdeposit\b)",
    re.IGNORECASE,
)
_TIMEFRAME_PATTERN = re.compile(
    r"\b(immediate|urgent|soon|day|week|month|year|explore|exploring|research|ready|now)\b",
    re.IGNORECASE,
)
_OPTION_REQUEST_PATTERN = re.compile(
    r"\b(option|options|choice|choices|kya option|show options|list options|give options|provide option|provide options|options batao)\b",
    re.IGNORECASE,
)
_SCHEDULING_PATTERN = re.compile(
    r"\b(schedule|meeting|calendly|consultation|appointment|book slot|book a slot|call slot|"
    r"talk to agent|speak with agent|site visit|site visit slot|site visit schedule|timing|"
    r"time slot|morning slot|afternoon slot|evening slot|morning|afternoon|evening|weekend visit|slot)\b",
    re.IGNORECASE,
)
_INAPPROPRIATE_PATTERN = re.compile(
    r"\b("
    r"fuck|fucking|fucker|shit|bitch|bastard|asshole|dick|cock|pussy|porn|nude|sex|sexy|horny|"
    r"idiot|stupid|dumb|shut\s*up|kill\s*yourself|useless|scam|fraud|cheat|loot|chor|"
    r"chutiya|chutiye|madarchod|bhosdike|bhosadi|gaand|gandu|harami|kameena|saale|saala|kutta|kutte|"
    r"bewakoof|pagal|bakwas|bakwaas|lund|lauda|randi|terimak|bc|mc"
    r")\b",
    re.IGNORECASE,
)
_GIBBERISH_PATTERN = re.compile(
    r"(\b(asdf|qwerty|zxcv|hjkl|dfgh)\w*\b|([a-zA-Z])\3{4,}|\b[bcdfghjklmnpqrstvwxyz]{5,}\b)",
    re.IGNORECASE,
)


def is_inappropriate_message(text: str) -> bool:
    """Detect offensive words, abusive language, insults, or random keyboard smash."""
    if not text:
        return False
    return bool(_INAPPROPRIATE_PATTERN.search(text) or _GIBBERISH_PATTERN.search(text))


def get_inappropriate_message_redirection(answers: dict, lang: str = "en") -> str:
    """Return a polite, professional boundary that immediately steers conversation back to property enquiry in English, Hindi, or Hinglish."""
    pending_slot = next((s for s in ORDERED_REQUIRED_SLOTS if not answers.get(s)), "location_preference")
    prompts = QUESTION_PROMPTS_MULTILINGUAL.get(lang, QUESTION_PROMPTS_MULTILINGUAL["en"])
    prefix = prompts.get("inappropriate", QUESTION_PROMPTS_MULTILINGUAL["en"]["inappropriate"])
    q_text = get_question_prompt(pending_slot, lang=lang)
    return prefix + q_text


def extract_all_slots_from_text(text: str, current_answers: dict) -> dict:
    """Robustly extract any property preferences mentioned in free-form user message."""
    extracted = {}
    txt_low = text.lower().strip()

    # 1. Location extraction (cities & localities)
    city = None
    if any(w in txt_low for w in ["haldwani", "हल्द्वानी"]):
        city = "Haldwani"
    elif any(w in txt_low for w in ["nainital", "नैनीताल"]) and any(w in txt_low for w in ["bhimtal", "भीमताल"]):
        city = "Nainital / Bhimtal"
    elif any(w in txt_low for w in ["nainital", "नैनीताल"]):
        city = "Nainital"
    elif any(w in txt_low for w in ["bhimtal", "भीमताल"]):
        city = "Bhimtal"
    elif any(w in txt_low for w in ["almora", "अल्मोड़ा", "अल्मोडा"]):
        city = "Almora"
    elif any(w in txt_low for w in ["bhowali", "भवाली"]):
        city = "Bhowali"
    elif any(w in txt_low for w in ["dehradun", "देहरादून"]):
        city = "Dehradun"
    elif any(w in txt_low for w in ["rishikesh", "ऋषिकेश"]):
        city = "Rishikesh"
    elif any(w in txt_low for w in ["haridwar", "हरिद्वार"]):
        city = "Haridwar"
    elif any(w in txt_low for w in ["mukteshwar", "मुक्तेश्वर"]):
        city = "Mukteshwar"
    elif any(w in txt_low for w in ["ranikhet", "रानीखेत"]):
        city = "Ranikhet"
    elif any(w in txt_low for w in ["ramnagar", "रामनगर"]):
        city = "Ramnagar"
    elif any(w in txt_low for w in ["rudrapur", "रुद्रपुर"]):
        city = "Rudrapur"
    elif any(w in txt_low for w in ["kashipur", "काशीपुर"]):
        city = "Kashipur"

    area = None
    for a in [
        "rampur road", "kaladhungi road", "kaladhungi", "bithoria", "mukhani", "kusumkhera", "kathgodam",
        "damuadhunga", "panchakki", "nawabi road", "heera nagar", "tikonia", "kamaluaganja",
        "mallital", "tallital", "pangot", "naukuchiatal", "sattal", "mehragaon", "kasar devi",
        "bright end corner", "ranikhet road", "rajpur road", "sahastradhara", "tapovan", "laxman jhula",
        "kainchi dham", "ramgarh"
    ]:
        if a in txt_low:
            area = a.title()
            if not city:
                if a in ["rampur road", "kaladhungi road", "kaladhungi", "bithoria", "mukhani", "kusumkhera", "kathgodam", "damuadhunga", "panchakki", "nawabi road", "heera nagar", "tikonia", "kamaluaganja"]:
                    city = "Haldwani"
                elif a in ["mallital", "tallital", "pangot"]:
                    city = "Nainital"
                elif a in ["naukuchiatal", "sattal", "mehragaon"]:
                    city = "Bhimtal"
                elif a in ["kasar devi", "bright end corner"]:
                    city = "Almora"
                elif a in ["rajpur road", "sahastradhara"]:
                    city = "Dehradun"
                elif a in ["tapovan", "laxman jhula"]:
                    city = "Rishikesh"
                elif a in ["kainchi dham", "ramgarh"]:
                    city = "Bhowali"
            break

    if city and area:
        extracted["location_preference"] = f"{city} ({area})"
        extracted["location_city"] = city.lower()
        extracted["location_area"] = area
    elif city:
        extracted["location_preference"] = city
        extracted["location_city"] = city.lower()
    elif area:
        extracted["location_preference"] = area
        extracted["location_area"] = area

    # 2. BHK / Configuration extraction
    bhk = extract_bhk_from_text(txt_low)
    if bhk:
        extracted["bhk_preference"] = bhk
    elif any(w in txt_low for w in ["plot", "land", "zameen", "ploting", "जमीन", "प्लॉट"]):
        extracted["bhk_preference"] = "Residential Plot / Land"
    elif any(w in txt_low for w in ["commercial", "shop", "office", "dukan", "दुकान", "शोरूम"]):
        extracted["bhk_preference"] = "Commercial Space / Shop"

    # 3. Budget extraction (numeric numbers, crores, lakhs, or standard ranges)
    # Check multi-number ranges first before matching single numbers
    if any(w in txt_low for w in ["above 2 crore", "over 2 crore", "2 crore+", "2 cr+", "2cr+"]):
        extracted["budget_range"] = "Above 2 crore"
    elif any(w in txt_low for w in ["1 crore - 2 crore", "1 to 2 crore", "1-2 cr", "1-2 crore", "1 to 2 cr", "1.5 cr", "1.5 crore", "1.2 cr", "1.8 cr"]):
        extracted["budget_range"] = "1 crore - 2 crore"
    elif any(w in txt_low for w in ["25 lakh - 50 lakh", "25 to 50 lakh", "25-50 lakh", "25 to 50", "25-50", "25 lac - 50 lac", "25-50 lac"]):
        extracted["budget_range"] = "25 lakh - 50 lakh"
    elif any(w in txt_low for w in [
        "50 lakh - 1 crore", "50 to 1 crore", "50-1 cr", "50-1 crore", "50 to 1 cr", "50l - 1cr", "50l-1cr",
        "50-100 lakh", "50 to 100 lakh", "50-100 lac"
    ]) or re.search(r"\b50\s*(?:-|to)\s*1\s*(?:cr|crore)\b", txt_low):
        extracted["budget_range"] = "50 lakh - 1 crore"
    elif any(w in txt_low for w in ["up to 25 lakh", "under 25 lakh", "below 25 lakh"]):
        extracted["budget_range"] = "Up to 25 lakh"
    # Single amounts / fallback regex
    elif any(w in txt_low for w in ["50 lakh", "60 lakh", "70 lakh", "80 lakh", "90 lakh", "75 lakh", "65 lakh", "85 lakh", "50 lac", "60 lac", "70 lac", "80 lac"]):
        extracted["budget_range"] = "50 lakh - 1 crore"
    elif any(w in txt_low for w in ["30 lakh", "35 lakh", "40 lakh", "45 lakh", "25 lakh", "30 lac", "35 lac", "40 lac", "45 lac", "25 lac"]):
        extracted["budget_range"] = "25 lakh - 50 lakh"
    elif any(w in txt_low for w in ["15 lakh", "20 lakh", "10 lakh"]):
        extracted["budget_range"] = "Up to 25 lakh"
    elif any(w in txt_low for w in ["1 cr", "1 crore", "2 crore", "2 cr"]):
        extracted["budget_range"] = "1 crore - 2 crore"
    else:
        num_match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|lacs|cr|crore|लाख|करोड़)", txt_low)
        if num_match:
            val = float(num_match.group(1))
            unit = num_match.group(2)
            if "cr" in unit or "करोड़" in unit:
                if val >= 2.0:
                    extracted["budget_range"] = "Above 2 crore"
                elif val >= 1.0:
                    extracted["budget_range"] = "1 crore - 2 crore"
                else:
                    extracted["budget_range"] = "50 lakh - 1 crore"
            else:
                if val < 25.0:
                    extracted["budget_range"] = "Up to 25 lakh"
                elif val <= 50.0:
                    extracted["budget_range"] = "25 lakh - 50 lakh"
                elif val <= 100.0:
                    extracted["budget_range"] = "50 lakh - 1 crore"
                elif val <= 200.0:
                    extracted["budget_range"] = "1 crore - 2 crore"
                else:
                    extracted["budget_range"] = "Above 2 crore"

    # 4. Timeline extraction
    if any(w in txt_low for w in ["immediately", "immediate", "urgent", "jaldi", "turant", "ready to move", "तुरंत"]):
        extracted["planning_to_buy"] = "Immediately"
    elif any(w in txt_low for w in ["within 1 month", "1 month", "one month", "next month", "1 mahina", "1 महीने"]):
        extracted["planning_to_buy"] = "Within 1 Month"
    elif any(w in txt_low for w in ["within 3 months", "3 months", "2 months", "1 to 3", "3 mahine", "3 महीने"]):
        extracted["planning_to_buy"] = "Within 3 Months"
    elif any(w in txt_low for w in ["within 6 months", "6 months", "3 to 6", "6 mahine", "6 महीने"]):
        extracted["planning_to_buy"] = "Within 6 Months"
    elif any(w in txt_low for w in ["within 12 months", "12 months", "1 year", "1 saal"]):
        extracted["planning_to_buy"] = "Within 12 Months"
    elif any(w in txt_low for w in ["just exploring", "exploring", "researching", "dekh rahe"]):
        extracted["planning_to_buy"] = "Just Exploring / Researching"

    # 5. Payment method extraction
    if any(w in txt_low for w in ["home loan", "mortgage", "bank finance", "bank loan", "emi", "loan", "लोन"]):
        extracted["payment_option"] = "Home Loan (Mortgage)"
    elif any(w in txt_low for w in ["full cash", "self funded", "own funds", "direct payment", "cash", "नकद", "payement"]):
        extracted["payment_option"] = "Full Cash Payment"
    elif any(w in txt_low for w in ["down payment", "part payment", "downpayment"]):
        extracted["payment_option"] = "Down Payment Plan"
    elif any(w in txt_low for w in ["installment", "clp", "construction linked", "किस्त"]):
        extracted["payment_option"] = "Construction-Linked Payment Plan (CLP)"

    # 6. Remarks / Special requirements extraction
    special_aspects = []
    if any(w in txt_low for w in ["school", "college", "kids", "children", "स्कूल"]):
        special_aspects.append("near reputed schools")
    if any(w in txt_low for w in ["lake view", "lake-view", "lake", "झील"]):
        special_aspects.append("lake view")
    if any(w in txt_low for w in ["mountain view", "hill view", "himalayan view", "पहाड़"]):
        special_aspects.append("mountain/hill view")
    if any(w in txt_low for w in ["parking", "car parking", "garage"]):
        special_aspects.append("dedicated parking")
    if any(w in txt_low for w in ["garden", "lawn", "greenery", "बगीचा"]):
        special_aspects.append("private garden/greenery")
    if any(w in txt_low for w in ["ground floor"]):
        special_aspects.append("ground floor preference")
    if any(w in txt_low for w in ["top floor", "penthouse floor"]):
        special_aspects.append("top floor preference")
    if any(w in txt_low for w in ["highway", "road facing", "main road"]):
        special_aspects.append("near main highway/road")
    if any(w in txt_low for w in ["quiet", "peaceful", "calm", "शांत"]):
        special_aspects.append("peaceful surroundings")
    if any(w in txt_low for w in ["hospital", "medical", "अस्पताल"]):
        special_aspects.append("near medical facilities")
    if any(w in txt_low for w in ["gated", "security"]):
        special_aspects.append("gated society with security")
    if any(w in txt_low for w in ["ready to move"]):
        special_aspects.append("ready-to-move unit")

    if special_aspects:
        extracted["remarks_requirements"] = ", ".join(special_aspects)

    return extracted


def generate_multi_slot_analysis_and_suggestions(answers: dict, new_slots: dict, lang: str = "en") -> str:
    """Generate comprehensive acknowledgement, market analysis, and property suggestions
    when customer provides multiple criteria at once."""
    loc = answers.get("location_preference") or answers.get("location_city", "").title() or "Haldwani"
    bhk = answers.get("bhk_preference") or ""
    budget = answers.get("budget_range") or ""
    payment = answers.get("payment_option") or ""
    timeline = answers.get("planning_to_buy") or ""
    remarks = answers.get("remarks_requirements") or ""

    if lang == "hi":
        bullets = []
        if loc: bullets.append(f"• पसंदीदा क्षेत्र: {loc}")
        if bhk: bullets.append(f"• प्रॉपर्टी कॉन्फ़िगरेशन: {bhk}")
        if budget: bullets.append(f"• बजट रेंज: {budget}")
        if payment: bullets.append(f"• भुगतान विकल्प: {payment}")
        if timeline: bullets.append(f"• समयसीमा: {timeline}")
        if remarks and remarks.lower() not in {"none", "skip"}: bullets.append(f"• विशेष आवश्यकता: {remarks}")
        summary_block = "शानदार चयन! हमने आपकी आवश्यकताएं नोट कर ली हैं:\n" + "\n".join(bullets)
    elif lang == "hinglish":
        bullets = []
        if loc: bullets.append(f"• Location: {loc}")
        if bhk: bullets.append(f"• Configuration: {bhk}")
        if budget: bullets.append(f"• Budget: {budget}")
        if payment: bullets.append(f"• Payment: {payment}")
        if timeline: bullets.append(f"• Timeline: {timeline}")
        if remarks and remarks.lower() not in {"none", "skip"}: bullets.append(f"• Special Requirement: {remarks}")
        summary_block = "Great choices! Humne aapki requirements note kar li hain:\n" + "\n".join(bullets)
    else:
        bullets = []
        if loc: bullets.append(f"• Location: {loc}")
        if bhk: bullets.append(f"• Configuration: {bhk}")
        if budget: bullets.append(f"• Budget: {budget}")
        if payment: bullets.append(f"• Payment Option: {payment}")
        if timeline: bullets.append(f"• Buying Timeline: {timeline}")
        if remarks and remarks.lower() not in {"none", "skip"}: bullets.append(f"• Special Requirement: {remarks}")
        summary_block = "Excellent choices! I have noted your preferences:\n" + "\n".join(bullets)

    insights = []
    loc_l = loc.lower()
    bhk_desc = bhk if bhk else "residential"

    if "haldwani" in loc_l:
        if lang == "hi":
            insights.append("हल्द्वानी में रामपुर रोड, कालाढूंगी रोड और मुखानी जैसे प्रमुख क्षेत्रों में आपके बजट और कॉन्फ़िगरेशन के अनुसार बेहतरीन गेटेड सोसाइटीज और बिल्डर फ्लोर्स उपलब्ध हैं।")
        elif lang == "hinglish":
            insights.append("Haldwani mein Rampur Road, Kaladhungi Road aur Mukhani jaise high-growth areas mein aapke budget aur configuration ke hisaab se premium gated societies aur builder floors available hain.")
        else:
            insights.append("In Haldwani, high-demand residential corridors like Rampur Road, Kaladhungi Road, and Mukhani offer verified gated societies and builder floors matching this profile.")
    elif any(w in loc_l for w in ["nainital", "bhimtal", "bhowali"]):
        if lang == "hi":
            insights.append("भीमताल व नैनीताल में झील और खूबसूरत वादियों के नज़दीक संपत्तियां छुट्टियां बिताने और हॉलिडे होम रेंटल के लिए सर्वोत्तम हैं।")
        elif lang == "hinglish":
            insights.append("Bhimtal aur Nainital mein lake-view aur scenic mountain-facing properties vacation living aur high-return rentals ke liye perfect hain.")
        else:
            insights.append("In the Bhimtal & Nainital region, properties with peaceful valley or lake views are in prime demand for serene living and holiday rentals.")
    elif "almora" in loc_l:
        if lang == "hi":
            insights.append("अल्मोड़ा में हिमालयी दृश्यों वाली संपत्तियां शांतिपूर्ण जीवन और दीर्घकालिक निवेश के लिए बहुत उपयुक्त हैं।")
        else:
            insights.append("Almora offers tranquil mountain-view retreat properties and peaceful living with great long-term appreciation.")

    if budget:
        if "50 lakh" in budget.lower() or "1 crore" in budget.lower():
            if lang == "hi":
                insights.append(f"{budget} के बजट में उत्तम सड़क कनेक्टिविटी, सुरक्षा और आवश्यक सुविधाओं वाली {bhk_desc} प्रॉपर्टीज तुरंत उपलब्ध हैं।")
            elif lang == "hinglish":
                insights.append(f"{budget} budget bracket mein prime road connectivity aur gated security wali {bhk_desc} properties best value deti hain.")
            else:
                insights.append(f"Within the {budget} range, {bhk_desc} options with prime road connectivity and clear title documentation are readily available.")
        elif "above 2 crore" in budget.lower():
            if lang == "hi":
                insights.append(f"{budget} के प्रीमियम बजट में निजी लॉन, खूबसूरत दृश्यों वाली लक्जरी इंडिपेंडेंट विला सबसे उत्तम विकल्प हैं।")
            else:
                insights.append(f"For your premium budget of {budget}, luxury independent villas and expansive foothill estates are ideal choices.")

    if payment:
        if "cash" in payment.lower():
            if lang == "hi":
                insights.append("पूरा नकद भुगतान होने से आपको तुरंत पजेशन, त्वरित रजिस्ट्री और डेवलपर्स से विशेष मूल्य लाभ मिलता है।")
            elif lang == "hinglish":
                insights.append("Full cash payment readiness aapko immediate possession, fast-track registration aur builder negotiations mein strong advantage deti hai.")
            else:
                insights.append("Your full cash readiness provides a strong advantage for immediate possession and negotiated builder pricing.")
        elif "loan" in payment.lower():
            if lang == "hi":
                insights.append("हम अग्रणी बैंकों (SBI, HDFC, ICICI) के साथ आसान लोन स्वीकृति और सरल दस्तावेज़ीकरण में पूरी सहायता करते हैं।")
            else:
                insights.append("We partner with leading banks to ensure smooth home loan approvals and hassle-free documentation.")

    if remarks and remarks.lower() not in {"none", "skip"}:
        if lang == "hi":
            insights.append(f"आपकी आवश्यकता ({remarks}) को हम शॉर्टलिस्ट की जाने वाली सभी प्रॉपर्टीज में प्राथमिकता देंगे।")
        elif lang == "hinglish":
            insights.append(f"Aapki specific requirement ({remarks}) ko humari team top priority pe shortlist karegi.")
        else:
            insights.append(f"We have specifically noted your preference for {remarks} and will ensure it is prioritized.")

    header = "💡 प्रॉपर्टी सुझाव व बाज़ार विश्लेषण:" if lang == "hi" else ("💡 Property Suggestions & Market Insights:" if lang == "en" else "💡 Property Suggestions & Market Insights:")
    advice_block = f"{header}\n" + " ".join(insights)
    return f"{summary_block}\n\n{advice_block}"


def generate_completion_lead_analysis(answers: dict, lang: str = "en") -> str:
    """Generate final comprehensive real estate analysis and recommendations before presenting
    the booking summary card and Calendly consultation links."""
    loc = answers.get("location_preference") or answers.get("location_city", "").title() or "Uttarakhand"
    bhk = answers.get("bhk_preference") or "property"
    budget = answers.get("budget_range") or ""
    payment = answers.get("payment_option") or ""
    timeline = answers.get("planning_to_buy") or ""
    remarks = answers.get("remarks_requirements") or ""

    insights = []
    loc_l = loc.lower()

    if "haldwani" in loc_l:
        if lang == "hi":
            insights.append(f"• हल्द्वानी के रामपुर रोड और कालाढूंगी रोड पर {budget} के बजट में {bhk} के गेटेड सोसाइटी और प्रीमियम फ्लोर्स उपलब्ध हैं।")
        elif lang == "hinglish":
            insights.append(f"• Haldwani mein Rampur Road aur Kaladhungi Road corridors pe {budget} budget ke mutabiq {bhk} verified options best suit karte hain.")
        else:
            insights.append(f"• In Haldwani, prime corridors along Rampur Road and Kaladhungi Road feature verified {bhk} societies matching your {budget} budget.")
    elif any(w in loc_l for w in ["nainital", "bhimtal", "bhowali"]):
        if lang == "hi":
            insights.append(f"• नैनीताल व भीमताल क्षेत्र में {budget} के बजट में शांत पहाड़ी और झील के नज़ारे वाली संपत्तियां छुट्टियां बिताने और रेंटल यील्ड के लिए उत्तम हैं।")
        else:
            insights.append(f"• In Bhimtal & Nainital, scenic valley and lake-facing {bhk} properties in the {budget} range offer premier living and holiday rental returns.")
    elif "almora" in loc_l:
        if lang == "hi":
            insights.append(f"• अल्मोड़ा में शांत पहाड़ी वातावरण और 180-डिग्री हिमालयी दृश्यों वाली संपत्तियां आपकी आवश्यकता के लिए सर्वोपरि हैं।")
        else:
            insights.append(f"• In Almora, tranquil mountain-view retreat properties offer peaceful living with high appreciation potential.")
    else:
        insights.append(f"• Verified {bhk} options in {loc} matching your {budget} range are being prioritized by our specialist team.")

    if payment:
        if "cash" in payment.lower():
            if lang == "hi":
                insights.append("• नकद भुगतान और आपकी समयसीमा के अनुसार, आपको रेडी-टू-मूव प्रॉपर्टीज पर तत्काल पजेशन और रजिस्ट्री का विशेष लाभ मिलेगा।")
            elif lang == "hinglish":
                insights.append("• Full cash readiness aur aapke timeline ke sath, immediate possession aur developer discounts par priority deal milegi.")
            else:
                insights.append("• Your full cash readiness and timeline provide direct leverage for immediate possession and negotiated developer pricing.")
        elif "loan" in payment.lower():
            if lang == "hi":
                insights.append("• बैंक लोन के लिए सभी स्वीकृत और स्पष्ट टाइटल वाली प्रॉपर्टीज को प्राथमिकता दी जाएगी।")
            else:
                insights.append("• Clear-title projects with pre-approved bank loans will be prioritized for smooth, hassle-free financing.")

    if remarks and remarks.lower() not in {"none", "skip"}:
        if lang == "hi":
            insights.append(f"• आपकी विशेष आवश्यकता ({remarks}) का पूरा ध्यान रखा जाएगा।")
        else:
            insights.append(f"• Dedicated attention will be given to your requirement for {remarks}.")

    header = "🌟 प्रोफिलेंट प्रॉपर्टी विश्लेषण व सुझाव (Market Insights):" if lang == "hi" else ("🌟 Profilent Property Analysis & Recommendations:" if lang == "en" else "🌟 Profilent Property Analysis & Recommendations:")
    return f"{header}\n" + "\n".join(insights)


def analyze_user_query_and_intent(msg: str, answers: dict, lang: str = "en") -> str | None:
    """Analyze what the user is asking or discussing and generate consultative real estate output."""
    msg_low = msg.lower().strip()

    # 1. School & Education query
    if any(w in msg_low for w in ["school", "college", "children", "kids", "admission", "padhai", "bacche", "स्कूल", "कॉलेज", "बच्चे"]):
        if lang == "hi":
            return (
                "हल्द्वानी में मुखानी और बिठौरिया परिवारों के लिए सबसे पसंदीदा इलाके हैं, जहाँ बिरला स्कूल, बीरशिबा और सिंथिया जैसे प्रतिष्ठित स्कूल चंद मिनटों की दूरी पर हैं! 🏫 "
                "यहाँ सुरक्षित आवासीय कॉलोनियाँ, पार्क और शांत वातावरण उपलब्ध है।"
            )
        elif lang == "hinglish":
            return (
                "Haldwani me Mukhani aur Bithoria families ke liye top localities hain, jahan Birla School, Beersheba aur top institutions bilkul pass hain! 🏫 "
                "Yeh safe residential societies, parks aur wide roads ke sath perfect family environment dete hain."
            )
        else:
            return (
                "For families with school-going children, Mukhani and Bithoria in Haldwani are premier localities! 🏫 "
                "Top institutions like Birla School, Beersheba, and Cynthia Senior Secondary are just minutes away, offering peaceful residential gated societies and parks."
            )

    # 2. Price / Rate enquiry
    if any(w in msg_low for w in ["price", "rate", "cost", "budget", "kitna kharcha", "rate kya hai", "kya rate", "daam", "रेट", "कीमत", "भाव"]):
        if any(w in msg_low for w in ["nainital", "bhimtal", "bhowali", "mukteshwar", "hill", "pahar"]):
            if lang == "hi":
                return (
                    "भीमताल और नैनीताल में माउंटेन-व्यू 1 व 2 BHK हॉलिडे अपार्टमेंट्स ₹45 लाख – ₹75 लाख से शुरू होते हैं, जबकि लेक-व्यू विला और स्वतंत्र कॉटेज ₹90 लाख – ₹1.8 करोड़ के बीच उपलब्ध हैं। 🏔️"
                )
            elif lang == "hinglish":
                return (
                    "Bhimtal aur Nainital me scenic mountain-view 1 & 2 BHK apartments ₹45 Lakh - ₹75 Lakh se start hote hain, jabki premium lake-view villas aur independent cottages ₹90 Lakh - ₹1.8 Crore tak available hain! 🏔️"
                )
            else:
                return (
                    "In Bhimtal and Nainital, scenic mountain-view 1 & 2 BHK holiday apartments start from ₹45 Lakh – ₹75 Lakh, while luxury lake-facing independent villas and cottages range between ₹90 Lakh – ₹1.8 Crore! 🏔️"
                )
        else:
            if lang == "hi":
                return (
                    "हल्द्वानी में 2 BHK फ्लैट्स ₹35 लाख – ₹55 लाख, 3 BHK अपार्टमेंट्स व स्वतंत्र मकान ₹65 लाख – ₹1.1 करोड़, तथा आवासीय प्लॉट्स ₹1,800 – ₹3,500 प्रति वर्ग फुट की दर पर उपलब्ध हैं। 🏡"
                )
            elif lang == "hinglish":
                return (
                    "Haldwani me modern 2 BHK flats ₹35 Lakh - ₹55 Lakh, spacious 3 BHK apartments & independent houses ₹65 Lakh - ₹1.1 Crore, aur residential plots ₹1,800 - ₹3,500/sq.ft ke budget me available hain! 🏡"
                )
            else:
                return (
                    "In Haldwani, modern 2 BHK flats typically range between ₹35 Lakh – ₹55 Lakh, 3 BHK luxury homes & independent houses between ₹65 Lakh – ₹1.1 Crore, and residential plots from ₹1,800 – ₹3,500 per sq. ft. 🏡"
                )

    # 3. Area comparison (Rampur Rd vs Kaladhungi Rd)
    if ("rampur" in msg_low and "kaladhungi" in msg_low) or ("difference" in msg_low and ("road" in msg_low or "area" in msg_low)):
        if lang == "hi":
            return (
                "रामपुर रोड मुख्य हाईवे व बाजारों से सीधी कनेक्टिविटी वाला तेजी से बढ़ता कमर्शियल व रिहायशी कॉरिडोर है। वहीं कालाढूंगी रोड शांत, हरा-भरा, प्रतिष्ठित स्कूलों के समीप और परिवारों के सुकून भरे जीवन के लिए आदर्श है! 🌳"
            )
        elif lang == "hinglish":
            return (
                "Rampur Road fast highway transit aur main market connectivity ke liye famous hai. Wahi Kaladhungi Road quiet, green surroundings aur premier schools ke sath peaceful family living ke liye best hai! 🌳"
            )
        else:
            return (
                "Rampur Road is a high-growth corridor with fast highway connectivity, transport hubs, and commercial markets. Kaladhungi Road offers a greener, quieter residential setting with top schools and peaceful family societies! 🌳"
            )

    # 4. Loan / Bank financing query
    if any(w in msg_low for w in ["loan", "emi", "bank", "finance", "kist", "ऋण", "लोन", "किस्त"]):
        if lang == "hi":
            return (
                "जी हाँ! हमारे सभी प्रोजेक्ट्स रेरा अप्रूव्ड हैं और प्रमुख बैंकों (SBI, HDFC, ICICI, Bank of Baroda) से 80% से 85% तक आसान होम लोन व सुविधाजनक EMI विकल्पों के साथ टाई-अप है! 🏦"
            )
        elif lang == "hinglish":
            return (
                "Haan bilkul! Hamare sabhi projects RERA approved hain aur leading nationalized banks (SBI, HDFC, ICICI) ke sath tie-up hai, jahan aapko 80-85% tak easy home loan aur comfortable EMIs milti hain! 🏦"
            )
        else:
            return (
                "Yes, absolutely! All our verified projects have clear RERA documentation and pre-approved tie-ups with leading banks (SBI, HDFC, ICICI, etc.) providing up to 80-85% home loans with competitive EMI options! 🏦"
            )

    # 5. Lake / Mountain view / Holiday retreat query
    if any(w in msg_low for w in ["lake view", "mountain view", "hill view", "valley view", "lake", "view", "nature", "weather", "झील", "पहाड़", "व्यू"]):
        if lang == "hi":
            return (
                "भीमताल, भवाली और मुक्तेश्वर की शांत वादियों में आपको हिमालय की बर्फ से ढकी चोटियों और झीलों के मनोरम दृश्य मिलते हैं - जो वेकेशन होम या रिटायरमेंट के लिए सर्वोत्तम हैं! 🌄"
            )
        elif lang == "hinglish":
            return (
                "Bhimtal, Bhowali aur Mukteshwar serene Himalayan peaks aur picturesque lake views ke sath holiday homes aur peaceful retirement retreats ke liye sabse best hain! 🌄"
            )
        else:
            return (
                "Bhimtal, Bhowali, and Mukteshwar offer breathtaking panoramic Himalayan and lake vistas with year-round crisp mountain air - ideal for peaceful retreats or luxury vacation cottages! 🌄"
            )

    # 6. Investment / Rental yield query
    if any(w in msg_low for w in ["investment", "rental", "income", "yield", "return", "airbnb", "homestay", "munafa", "निवेश", "कमाई"]):
        if lang == "hi":
            return (
                "उत्तराखंड की तलहटी और हिल स्टेशनों में पर्यटन के चलते होमस्टे व वेकेशन रेंटल पर 8% से 12% तक का आकर्षक वार्षिक रिटर्न मिल रहा है, साथ ही हाईवे विस्तार से प्रॉपर्टी वैल्यू तेजी से बढ़ रही है! 📈"
            )
        elif lang == "hinglish":
            return (
                "Uttarakhand foothills aur hill stations me heavy tourism demand ke chalte homestays & vacation rentals pe 8% - 12% tak rental yield milti hai, plus upcoming highway connectivity se high capital growth hai! 📈"
            )
        else:
            return (
                "Uttarakhand's foothill belt and tourist hubs offer strong 8%–12% annual rental yields through managed vacation homestays and Airbnb, alongside rapid capital appreciation driven by major highway upgrades! 📈"
            )

    # 7. Office location or Visit enquiry
    if any(w in msg_low for w in ["office", "location", "kahan hai", "address", "visit", "milna", "ऑफिस", "पता"]):
        if lang == "hi":
            return (
                "हमारा मुख्य क्षेत्रीय प्रॉपर्टी एडवाइजरी ऑफिस हल्द्वानी, उत्तराखंड में स्थित है। हमारी टीम सातों दिन साइट विजिट और व्यक्तिगत परामर्श के लिए उपलब्ध है! 📍"
            )
        elif lang == "hinglish":
            return (
                "Hamara regional advisory office Haldwani, Uttarakhand me located hai. Hamari expert team site visits aur detailed property consultation ke liye 7 days available rehti hai! 📍"
            )
        else:
            return (
                "Our regional property advisory office is based in Haldwani, Uttarakhand. Our specialist team is available 7 days a week for in-person property consultations and guided site visits across the region! 📍"
            )

    # 8. Commercial / Retail / Plots
    if any(w in msg_low for w in ["commercial", "shop", "office space", "dukan", "showroom", "clinic", "दुकान", "कमर्शियल"]):
        if lang == "hi":
            return (
                "हमारे पास हल्द्वानी के मुख्य हाईवे और प्रमुख चौराहों पर प्राइम कमर्शियल दुकानें, शोरूम और ऑफिस स्पेस उपलब्ध हैं, जहाँ शानदार फुटफॉल और हाई रेंटल रिटर्न मिलता है! 🏢"
            )
        elif lang == "hinglish":
            return (
                "Haldwani ke prime highway corridors aur central commercial hubs me high-footfall retail shops, showrooms aur office spaces available hain with great rental returns! 🏢"
            )
        else:
            return (
                "We offer prime commercial retail shops, highway showrooms, and modern office spaces in Haldwani's highest footfall business corridors, ideal for high business growth and steady rental returns! 🏢"
            )

    return None


def _sanitize_and_validate_turn(
    extracted: dict,
    answers: dict,
    user_message: str,
    llm_reply: str,
    llm_complete: bool,
    was_already_complete: bool = False,
    recent_messages: list | None = None,
    is_real_llm_reply: bool = False,
) -> tuple[dict, str, bool]:
    """Take ANY kind of input, analyze intent and slots, and give contextual consultative output."""
    # 0. Track and persist language
    current_lang = answers.get("language", "en")
    lang = detect_language(user_message, current_lang=current_lang)
    answers["language"] = lang

    # Strip repetitive greetings on subsequent turns
    has_prior_history = bool(recent_messages and len(recent_messages) > 1)
    if has_prior_history or was_already_complete:
        llm_reply = _strip_repetitive_greetings(llm_reply)

    all_6_slots_present_before = all(
        bool(answers.get(s))
        for s in ["location_preference", "planning_to_buy", "budget_range", "payment_option", "bhk_preference", "remarks_requirements"]
    )
    if was_already_complete and all_6_slots_present_before:
        consult_answer = analyze_user_query_and_intent(user_message, answers, lang=lang)
        reply = llm_reply or consult_answer or "Our property specialist is reviewing your request and will reach out with customized options."
        return answers, reply, False

    # 1. Check if customer asked to speak to a person or human executive
    if _SPEAK_TO_HUMAN_PATTERN.search(user_message):
        if not answers.get("remarks_requirements"):
            answers["remarks_requirements"] = "Customer requested to speak with a sales executive"
        missing_core = [s for s in ORDERED_REQUIRED_SLOTS if not answers.get(s)]
        if not missing_core:
            return answers, "", True
        next_slot = missing_core[0]
        reassurance = {
            "hi": "हमारे सेल्स एग्जीक्यूटिव जल्द ही आपको कॉल करेंगे। वे आपके लिए सही प्रॉपर्टी ला सकें, इसके लिए कृपया बताएं:\n\n",
            "hinglish": "Hamare sales executive jald hi aapko call karenge. Woh aapke hisab se best options laa sakein, iske liye please batayein:\n\n",
            "en": "Our sales executive will connect with you shortly! To help them prepare the best options for you, please let us know:\n\n",
        }.get(lang, "Our sales executive will connect with you shortly! To help them prepare the best options for you, please let us know:\n\n")
        return answers, f"{reassurance}{get_question_prompt(next_slot, lang=lang)}", False

    last_answered_slot = None
    active_questions = get_active_questions_sync()
    all_active_keys = [q["slot_key"] for q in active_questions if q.get("slot_key")]
    slots_before = {k: answers.get(k) for k in all_active_keys if answers.get(k)}

    # Determine which question was expecting an answer before this user message
    expected_slot = None
    expected_q = None
    for q in active_questions:
        sk = q.get("slot_key")
        if sk and not answers.get(sk):
            expected_slot = sk
            expected_q = q
            break

    # 2. Extract standard property slots directly from free-form user message
    freeform_slots = extract_all_slots_from_text(user_message, answers)
    for k, v in freeform_slots.items():
        if v and not answers.get(k):
            answers[k] = v
            last_answered_slot = k
        elif v and k in ["bhk_preference", "budget_range", "planning_to_buy", "payment_option"]:
            answers[k] = v
            last_answered_slot = k

    # 3. Check if user is selecting an area for an already chosen city (e.g. typing "3" or "Bithoria")
    if answers.get("location_city") and not answers.get("location_area"):
        city_k = answers["location_city"]
        city_t = answers.get("location_city_title") or city_k.title()
        dyn_opts = answers.get("dynamic_locality_options")
        is_loc, chosen_area = resolve_locality_choice(user_message, city_k, dynamic_options=dyn_opts)
        if is_loc:
            answers["location_area"] = chosen_area
            if chosen_area.strip().lower() == city_t.strip().lower():
                answers["location_preference"] = city_t
            else:
                answers["location_preference"] = f"{city_t} ({chosen_area})"
            last_answered_slot = "location_preference"

    # 4. Also merge any slots returned by LLM (Gemini)
    for key, val in (extracted or {}).items():
        if not val or not str(val).strip():
            continue
        clean_val = str(val).strip()
        if clean_val.lower() in {"not specified", "none", "unknown", "n/a", "null"}:
            continue
        if not answers.get(key):
            answers[key] = clean_val
            if not last_answered_slot:
                last_answered_slot = key

    # 5. Handle user response for the expected question slot (standard or custom admin question)
    if expected_slot and not answers.get(expected_slot):
        # 5a. Numbered menu choice (1, 2, 3...)
        choice = _resolve_numbered_choice(user_message, expected_slot)
        if choice:
            answers[expected_slot] = choice
            last_answered_slot = expected_slot
        else:
            # 5b. Match option text from the question's predefined options
            q_opts = (expected_q.get("options") or []) if expected_q else []
            matched_opt = None
            if q_opts:
                msg_clean = user_message.strip().lower()
                for opt in q_opts:
                    opt_str = str(opt).strip()
                    if msg_clean == opt_str.lower() or opt_str.lower() in msg_clean:
                        matched_opt = opt_str
                        break
            if matched_opt:
                answers[expected_slot] = matched_opt
                last_answered_slot = expected_slot
            else:
                # 5c. Free-form text reply (custom feedback, remarks, or any text answer)
                raw_msg = user_message.strip()
                if raw_msg.lower() in {"skip", "chhod do", "aage badho", "no", "none", "nahi"}:
                    answers[expected_slot] = answers.get("form_requirement") or "None"
                    last_answered_slot = expected_slot
                elif raw_msg:
                    answers[expected_slot] = raw_msg
                    last_answered_slot = expected_slot

    slots_after = {k: answers.get(k) for k in all_active_keys if answers.get(k)}
    newly_provided_slots = {k: v for k, v in slots_after.items() if k not in slots_before}

    # 6. Check if user asked a consultative question or query (Schools, Prices, Comparisons, Loans, etc.)
    consultative_ans = analyze_user_query_and_intent(user_message, answers, lang=lang)

    # 7. Check if LLM produced an intelligent, non-generic reply
    is_good_llm_reply = bool(
        is_real_llm_reply
        and llm_reply
        and len(llm_reply.strip()) > 30
        and not llm_reply.strip().startswith("{")
        and not any(err in llm_reply.lower() for err in ["error", "exception", "failed"])
    )

    # 8. Locality area refinement check
    area_prompt = None
    if answers.get("location_city") and not answers.get("location_area"):
        loc_data = get_locality_data_or_ai_suggestions(answers["location_city"], lang=lang)
        if loc_data and loc_data.get("question"):
            area_prompt = loc_data["question"].get(lang, loc_data["question"]["en"])

    # 9. Find remaining unanswered questions
    remaining_unanswered = [q["slot_key"] for q in active_questions if q.get("slot_key") and not answers.get(q["slot_key"])]

    # If all criteria are now answered -> Finish!
    if not remaining_unanswered:
        ack = _generate_assistant_acknowledgement(last_answered_slot, answers.get(last_answered_slot, ""), lang=lang) if last_answered_slot else ""
        return answers, ack, True

    # If locality area is pending:
    if area_prompt and "location_preference" not in remaining_unanswered:
        advisory = consultative_ans or (_generate_assistant_acknowledgement(last_answered_slot, answers.get(last_answered_slot, ""), lang=lang) if last_answered_slot else "")
        reply_to_send = f"{advisory}\n\n{area_prompt}" if advisory else area_prompt
        return answers, reply_to_send, False

    # Prompt the next unanswered question in order
    next_slot = remaining_unanswered[0]
    q_prompt = get_question_prompt(next_slot, lang=lang)
    if consultative_ans:
        return answers, f"{consultative_ans}\n\n{q_prompt}", False
    if is_good_llm_reply and any(opt_indicator in llm_reply for opt_indicator in ["1.", "1️⃣", "•", "?", "which", "what", "budget", "area"]):
        return answers, llm_reply, False
    if len(newly_provided_slots) >= 2:
        advisory = generate_multi_slot_analysis_and_suggestions(answers, newly_provided_slots, lang=lang)
    else:
        advisory = _generate_assistant_acknowledgement(last_answered_slot, answers.get(last_answered_slot, ""), lang=lang) if last_answered_slot else ""
    reply_to_send = f"{advisory}\n\n{q_prompt}" if advisory else q_prompt
    return answers, reply_to_send, False


def execute_hybrid_ai_turn(
    conversation: dict,
    user_message: str,
    phone: str,
    schedule_url: str,
) -> dict | None:
    """Run an AI turn, merge newly extracted slots, and return the reply and completion state."""
    answers = dict(conversation.get("answers", {}))
    messages = list(conversation.get("messages", []))
    current_lang = answers.get("language", "en")
    lang = detect_language(user_message, current_lang=current_lang)
    answers["language"] = lang
    had_remarks_before = bool(answers.get("remarks_requirements"))
    has_all_6_slots_before = all(
        bool(answers.get(k))
        for k in ["location_preference", "planning_to_buy", "budget_range", "payment_option", "bhk_preference", "remarks_requirements"]
    )
    had_all_core_slots_before = all(
        bool(answers.get(k))
        for k in ["location_preference", "planning_to_buy", "budget_range", "payment_option", "bhk_preference"]
    )
    # A conversation was already complete ONLY IF all 6 criteria (including remarks) were already present before!
    was_already_complete = (
        has_all_6_slots_before
        and (
            conversation.get("status") == "complete"
            or conversation.get("stage") == "complete"
            or bool(conversation.get("completed_at"))
        )
    )
    contact_info = {
        "full_name": answers.get("full_name") or conversation.get("full_name", ""),
        "first_name": answers.get("first_name") or conversation.get("first_name", ""),
        "last_name": answers.get("last_name") or conversation.get("last_name", ""),
        "email": answers.get("email") or conversation.get("email", ""),
        "phone": phone,
    }
    if not answers.get("email") and contact_info.get("email"):
        answers["email"] = contact_info["email"]
    if not answers.get("full_name") and contact_info.get("full_name"):
        answers["full_name"] = contact_info["full_name"]
    if not answers.get("first_name") and contact_info.get("first_name"):
        answers["first_name"] = contact_info["first_name"]
    if not answers.get("last_name") and contact_info.get("last_name"):
        answers["last_name"] = contact_info["last_name"]

    cal_ctx = {
        **answers,
        "full_name": answers.get("full_name") or contact_info.get("full_name", ""),
        "first_name": answers.get("first_name") or contact_info.get("first_name", ""),
        "last_name": answers.get("last_name") or (answers.get("full_name", "").split()[1] if len(answers.get("full_name", "").split()) > 1 else ""),
        "email": answers.get("email") or contact_info.get("email", ""),
        "phone": phone,
        "mobile_number": phone,
    }

    raw_meeting_url = (schedule_url or os.getenv("CALENDLY_URL", "https://calendly.com/shailendra-profilent/30min")).strip()
    raw_visit_url = os.getenv("CALENDLY_VISIT_URL", "https://calendly.com/rudrasinghdeo369/30min").strip()

    meeting_cal_url = build_prefilled_calendly_url(raw_meeting_url, cal_ctx)
    visit_cal_url = build_prefilled_calendly_url(raw_visit_url, cal_ctx)

    cal_url = meeting_cal_url

    # Direct handler if customer sends an inappropriate or abusive message: steer back to property enquiry
    if is_inappropriate_message(user_message):
        redirect_reply = get_inappropriate_message_redirection(answers, lang=lang)
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "reply_text": redirect_reply,
            "updated_answers": answers,
            "updated_messages": messages + [
                {"role": "user", "content": user_message, "timestamp": now_iso},
                {"role": "assistant", "content": redirect_reply, "timestamp": now_iso},
            ],
            "is_complete": False,
            "is_followup": was_already_complete,
        }

    # Direct handler if customer specifically asks to schedule meeting / consultation / Calendly
    if _SCHEDULING_PATTERN.search(user_message):
        schedule_reply = _get_calendly_schedule_block(
            cal_url=cal_url,
            answers=answers,
            phone=phone,
            lang=answers.get("language", lang),
        )
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "reply_text": schedule_reply,
            "updated_answers": answers,
            "updated_messages": messages + [
                {"role": "user", "content": user_message, "timestamp": now_iso},
                {"role": "assistant", "content": schedule_reply, "timestamp": now_iso},
            ],
            "is_complete": False,
            "is_followup": was_already_complete,
        }

    ai_result = call_gemini_agent(
        user_message,
        answers,
        messages,
        contact_info,
        is_followup=was_already_complete,
        schedule_url=meeting_cal_url,
        visit_url=visit_cal_url,
    )
    if not ai_result:
        first_name = answers.get("first_name") or "there"
        msg_l = user_message.lower()
        raw_extracted = {}
        if any(w in msg_l for w in ["job", "naukri", "relocation", "transfer", "interview"]):
            raw_reply = (
                "Congratulations on your job opportunity in Haldwani! 🌟\n\n"
                "Having a comfortable home close to your workplace makes daily commute so much easier. "
                "Here are great living options near commercial & office corridors in Haldwani:\n\n"
                "• 1 BHK Rental Apartment (₹6,000 - ₹12,000 / mo)\n"
                "• 2 BHK Family Rental Flat (₹12,000 - ₹20,000 / mo)\n"
                "• Independent House / Villa for Rent or Purchase\n"
                "• Residential Plot to build your own dream home\n\n"
                "Which area is your workplace located in, and which option works best for you?"
            )
            raw_complete = False
        elif any(w in msg_l for w in ["business", "shop", "dukan", "commercial", "startup", "cafe", "restaurant", "clinic"]):
            raw_reply = (
                "Wishing you immense success with your commercial venture! 🚀\n\n"
                "Prime location and footfall are key for high growth. Here are top commercial property options in Haldwani:\n\n"
                "• High-Street Retail Shop on Main Corridor\n"
                "• Modern Commercial Office Space in Business Hub\n"
                "• Highway Commercial Plot / Showroom Space\n\n"
                "Which configuration and budget do you prefer to explore?"
            )
            raw_complete = False
        elif any(w in msg_l for w in ["school", "college", "children", "kids", "admission", "tuition"]):
            raw_reply = (
                "Living near top schools saves precious travel time for your family every day. 📚\n\n"
                "Here are premier residential options in safe, family-friendly neighborhoods:\n\n"
                "• 2 BHK / 3 BHK Gated Society Apartment\n"
                "• Independent 3+ BHK Family Villa with Private Garden\n"
                "• Residential Plots in Premium Localities\n\n"
                "Which of these options would you prefer for your family?"
            )
            raw_complete = False
        elif any(w in msg_l for w in ["weather", "mausam", "tourism", "visit", "trip", "nainital", "bhimtal", "holiday", "peace"]):
            raw_reply = (
                "The foothills offer scenic Himalayan views and refreshing weather all year round. 🌄\n\n"
                "Here are popular peaceful getaway properties and holiday options:\n\n"
                "• Scenic Mountain-View Residential Plots in Bhimtal / Nainital\n"
                "• Luxury Holiday Villas in Peaceful Foothill Localities\n"
                "• Modern 2 BHK / 3 BHK Apartments with Beautiful Valley Views\n\n"
                "Are you considering a holiday retreat, retirement home, or an investment plot in the hills?"
            )
            raw_complete = False
        elif any(w in msg_l for w in ["house", "villa", "makan", "ghar"]):
            raw_extracted["bhk_preference"] = "Independent House / Villa"
            raw_reply = (
                "I've noted down your preference for an Independent House / Villa in Haldwani / Uttarakhand. 🏡\n\n"
                "Here are popular areas with premium independent villas:\n\n"
                "• Rampur Road (Prime connectivity & markets)\n"
                "• Kaladhungi Road (Peaceful residential hub)\n"
                "• Bithoria / Mukhani (Premium family locality)\n"
                "• Bhimtal / Nainital Road (Scenic foothill environment)\n\n"
                "Which area or budget range would you like us to shortlist for you?"
            )
            raw_complete = False
        elif _SCHEDULING_PATTERN.search(user_message):
            raw_reply = _get_calendly_schedule_block(
                cal_url=cal_url,
                answers=answers,
                phone=phone,
                lang=answers.get("language", lang),
            )
            raw_complete = False
        elif any(w in msg_l for w in ["option", "options", "suggest", "choice", "help", "de do", "bataye"]):
            first_missing = next(
                (k for k in ["bhk_preference", "budget_range", "planning_to_buy", "payment_option"] if not answers.get(k)),
                "bhk_preference"
            )
            option_texts = {
                "bhk_preference": "Here are our available property configurations:\n• 1 BHK Apartment\n• 2 BHK Apartment\n• 3 BHK Apartment\n• 4+ BHK Luxury Villa\n• Residential Plot / Land\n• Commercial Office / Retail Shop",
                "budget_range": "Here are standard budget ranges:\n• Up to 25 Lakh\n• 25 - 50 Lakh\n• 50 Lakh - 1 Crore\n• 1 - 2 Crore\n• Above 2 Crore\n(Or for rentals: • ₹5,000 - ₹15,000 / mo • ₹15,000 - ₹30,000 / mo)",
                "planning_to_buy": "When are you planning to finalize?\n• Immediately\n• Within 1 Month\n• Within 3 Months\n• Within 6 Months\n• Just Exploring",
                "payment_option": "What is your preferred payment structure?\n• Bank Finance / Home Loan with EMI\n• Full Cash Payment / Self Funding\n• Flexible Down Payment Plan\n• Standard Rental Agreement",
            }
            raw_reply = "Here are the available options:\n\n" + option_texts.get(first_missing, option_texts["bhk_preference"])
            raw_complete = False
        elif was_already_complete:
            raw_reply = (
                "Understood. 🏡 Here are popular property choices currently available in Haldwani & Uttarakhand:\n\n"
                "• 1 BHK / 2 BHK / 3 BHK Apartments\n"
                "• Luxury Independent Villas\n"
                "• Gated Residential Plots\n"
                "• Commercial Showrooms & Offices\n\n"
                "Which of these would you like to explore?"
            )
            raw_complete = False
        else:
            raw_reply = (
                "Here are the available property options in Haldwani & Uttarakhand:\n\n"
                "• 1 BHK / 2 BHK Modern Apartment\n"
                "• 3 BHK / 4 BHK Luxury Villa\n"
                "• Residential Plot / Land\n"
                "• Commercial Office / Retail Space\n\n"
                "Which configuration fits your requirement?"
            )
            raw_complete = False
    else:
        raw_extracted = ai_result.get("extracted_slots", {})
        raw_reply = ai_result.get("reply_message", "").strip()
        raw_complete = bool(ai_result.get("is_complete", False))

    answers, reply_to_send, is_complete = _sanitize_and_validate_turn(
        raw_extracted,
        answers,
        user_message,
        raw_reply,
        raw_complete,
        was_already_complete=was_already_complete,
        recent_messages=messages,
        is_real_llm_reply=bool(ai_result),
    )

    all_active_slots = [q["slot_key"] for q in get_active_questions_sync() if q.get("slot_key")]
    has_all_slots = all(bool(answers.get(s)) for s in all_active_slots)
    is_turn_completing = bool(
        not was_already_complete
        and (is_complete or has_all_slots)
    )

    consultative_ans = analyze_user_query_and_intent(user_message, answers, lang=lang)
    is_good_llm_reply = bool(
        raw_reply
        and len(raw_reply.strip()) > 30
        and not raw_reply.strip().startswith("{")
        and not any(err in raw_reply.lower() for err in ["error", "exception", "failed"])
    )

    if is_turn_completing:
        first_name = answers.get("first_name") or "there"
        spoke_person = bool(_SPEAK_TO_HUMAN_PATTERN.search(user_message))
        analysis_suggestions = generate_completion_lead_analysis(answers, lang=answers.get("language", lang))
        summary_msg = format_completion_summary_message(
            {**answers, "phone": phone, "mobile_number": answers.get("mobile_number") or phone},
            first_name=first_name,
            schedule_url=meeting_cal_url,
            spoke_to_person=spoke_person,
            lang=answers.get("language", lang),
        )
        parts = []
        if reply_to_send and reply_to_send.strip():
            parts.append(reply_to_send.strip())
        if consultative_ans:
            parts.append(consultative_ans)
        if analysis_suggestions:
            parts.append(analysis_suggestions)
        parts.append(summary_msg)
        reply_to_send = "\n\n".join(parts)
        is_complete = True
    elif was_already_complete:
        if consultative_ans:
            reply_to_send = consultative_ans
            if "calendly.com" not in reply_to_send:
                cal_block = _get_calendly_schedule_block(cal_url=meeting_cal_url, answers=cal_ctx, phone=phone, lang=answers.get("language", lang))
                reply_to_send = f"{reply_to_send.rstrip()}\n\n{cal_block}"
        elif is_good_llm_reply:
            if "calendly.com" not in reply_to_send:
                cal_block = _get_calendly_schedule_block(cal_url=meeting_cal_url, answers=cal_ctx, phone=phone, lang=answers.get("language", lang))
                reply_to_send = f"{reply_to_send.rstrip()}\n\n{cal_block}"
        else:
            u_low = user_message.strip().lower()
            is_explicit_summary_req = any(w in u_low for w in ["summary", "mera form", "my enquiry", "enquiry summary", "show summary"])
            if is_explicit_summary_req:
                first_name = answers.get("first_name") or "there"
                spoke_person = bool(_SPEAK_TO_HUMAN_PATTERN.search(user_message))
                reply_to_send = format_completion_summary_message(
                    {**answers, "phone": phone, "mobile_number": answers.get("mobile_number") or phone},
                    first_name=first_name,
                    schedule_url=meeting_cal_url,
                    spoke_to_person=spoke_person,
                    lang=answers.get("language", lang),
                )
            else:
                loc = answers.get("location_preference") or "Uttarakhand"
                bhk = answers.get("bhk_preference") or "property"
                reply_to_send = (
                    f"Our specialist is currently curating the top {bhk} options for you in {loc}. 🏡\n\n"
                    f"Feel free to ask any specific questions about localities, pricing, or RERA approvals, or book a quick consultation call below!"
                )
                cal_block = _get_calendly_schedule_block(cal_url=meeting_cal_url, answers=cal_ctx, phone=phone, lang=answers.get("language", lang))
                reply_to_send = f"{reply_to_send}\n\n{cal_block}"

    # Ensure any stray Calendly URLs in final reply text are completely prefilled
    reply_to_send = _prefill_all_calendly_urls_in_text(reply_to_send, cal_ctx)

    # Update message history
    now_iso = datetime.now(timezone.utc).isoformat()
    updated_messages = messages + [
        {"role": "user", "content": user_message, "timestamp": now_iso},
        {"role": "assistant", "content": reply_to_send, "timestamp": now_iso},
    ]

    return {
        "reply_text": reply_to_send,
        "updated_answers": answers,
        "updated_messages": updated_messages,
        "is_complete": is_complete and not was_already_complete,
        "is_followup": was_already_complete,
    }


def analyze_lead_with_ai(lead_data: dict, recent_messages: list | None = None) -> dict:
    """Use Google Gemini to analyze completed WhatsApp questionnaire data and qualify as Hot, Warm, or Cold.
    
    If AI classification succeeds:
    - Returns structured AI analysis with rating ('Hot', 'Warm', or 'Cold'), score (0-100), is_hot (bool), summary, and triggers.
    If AI is unavailable or fails:
    - Smoothly falls back to rule-based qualification engine from zoho_crm.evaluate_lead_quality.
    """
    from zoho_crm import evaluate_lead_quality

    wa_ans = lead_data.get("whatsapp_answers") or {}
    lead_summary = {
        "customer_name": lead_data.get("full_name") or f"{lead_data.get('first_name', '')} {lead_data.get('last_name', '')}".strip() or "Customer",
        "phone": lead_data.get("phone") or lead_data.get("mobile_number") or lead_data.get("contact_number") or "",
        "email": lead_data.get("email") or "",
        "location": lead_data.get("location_preference") or lead_data.get("location") or wa_ans.get("location_preference") or "",
        "configuration": lead_data.get("bhk_preference") or lead_data.get("bhk") or wa_ans.get("bhk_preference") or "",
        "budget": lead_data.get("budget_range") or lead_data.get("budget") or wa_ans.get("budget_range") or "",
        "timeline": lead_data.get("planning_to_buy") or lead_data.get("timeline") or wa_ans.get("planning_to_buy") or "",
        "payment": lead_data.get("payment_option") or lead_data.get("payment") or wa_ans.get("payment_option") or "",
        "requirements": lead_data.get("remarks_requirements") or lead_data.get("requirement") or lead_data.get("requirements") or lead_data.get("message") or wa_ans.get("remarks_requirements") or "",
        "calendly_booking": bool(lead_data.get("calendly_booking") or lead_data.get("calendly_event_name")),
    }

    conv_context = []
    if recent_messages:
        for m in recent_messages[-4:]:
            conv_context.append(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}")
    if conv_context:
        lead_summary["recent_chat"] = conv_context

    system_prompt = (
        "You are an expert Real Estate Lead Qualification Specialist for Profilent CRM.\n"
        "Analyze the customer's property enquiry to classify their buying urgency and lead quality as Hot, Warm, or Cold.\n\n"
        "Classification Rules:\n"
        "1. HOT LEAD (Score: 60 - 100, is_hot: true):\n"
        "   - Urgent or near-term buying timeline: 'Immediately', 'Within 1 Month', or 'Within 3 Months'\n"
        "   - Strong budget (50 lakh - 1 crore, 1 crore - 2 crore, Above 2 crore) with ready intent, cash, or home loan/mortgage\n"
        "   - Clear property configuration (e.g. 2 BHK, 2.5 BHK, 3 BHK, Villa) with active purchasing intent\n"
        "   - Direct consultation booking on Calendly or completed property questionnaire\n\n"
        "2. WARM LEAD (Score: 35 - 59, is_hot: false):\n"
        "   - Medium timeline: 'Within 6 Months'\n"
        "   - Moderate budget or exploratory purchasing intent beyond 3 months\n\n"
        "3. COLD LEAD (Score: 0 - 34, is_hot: false):\n"
        "   - Distant timeline: 'Within 12 Months', 'More Than 1 Year', or 'Just Exploring / Researching'\n"
        "   - Low budget or vague interest with no near-term purchasing intent\n\n"
        "Respond ONLY with valid JSON:\n"
        "{\n"
        '  "rating": "Hot" | "Warm" | "Cold",\n'
        '  "score": <integer 0-100>,\n'
        '  "is_hot": <boolean: true if Hot else false>,\n'
        '  "summary": "<1-2 sentence executive assessment of lead urgency and intent>",\n'
        '  "triggers": ["<list of 1-3 key qualification trigger points>"]\n'
        "}"
    )

    # 1. Try Gemini
    gemini_key, gemini_model = _get_gemini_config()
    if gemini_key:
        models_to_try = _get_candidate_models(gemini_model)

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": f"Customer lead details:\n{json.dumps(lead_summary, indent=2)}"}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 400,
            },
        }

        for current_model in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={gemini_key}"
                resp = _ai_session.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=(2.5, 4.5))
                if resp.status_code == 429:
                    _mark_model_exhausted(current_model, 600.0)
                    continue
                if not resp.ok:
                    continue
                body = resp.json()
                raw_text = body.get("candidates", [])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                parsed = _extract_json_object(raw_text)
                if not parsed:
                    continue
                rating = parsed.get("rating", "Warm").capitalize()
                if rating not in ("Hot", "Warm", "Cold"):
                    rating = "Hot" if parsed.get("is_hot") else "Warm"
                is_hot = (rating == "Hot")
                score = int(parsed.get("score") or (80 if is_hot else (45 if rating == "Warm" else 15)))

                # Cross-validate with CRM Master Qualification engine to prevent rating mismatch
                base_eval = evaluate_lead_quality(lead_data)
                if base_eval.get("is_hot") or base_eval.get("rating") == "Hot":
                    rating = "Hot"
                    is_hot = True
                    score = max(score, int(base_eval.get("score") or 85))

                summary = str(parsed.get("summary") or f"AI evaluated as {rating} lead")
                if is_hot and "warm" in summary.lower():
                    reasons_text = ", ".join(base_eval.get("reasons", []))
                    summary = f"Qualified as Hot Lead ({score}/100). {reasons_text}" if reasons_text else f"Qualified as Hot Lead ({score}/100)"

                triggers = parsed.get("triggers") or []
                if is_hot and not any("hot" in t.lower() or "urgent" in t.lower() for t in triggers):
                    triggers.extend(base_eval.get("triggers", []))

                return {
                    "rating": rating,
                    "score": score,
                    "is_hot": is_hot,
                    "summary": summary,
                    "triggers": triggers or [f"AI classified {rating}"],
                    "ai_qualified": True,
                }
            except Exception as e:
                logger.info("Gemini model %s AI lead analysis retry: %s", current_model, e)
                continue

    # Fallback to rule-based qualification if Gemini is not available or fails
    base_eval = evaluate_lead_quality(lead_data)
    return {
        "rating": base_eval["rating"],
        "score": base_eval["score"],
        "is_hot": base_eval["is_hot"],
        "summary": ", ".join(base_eval["reasons"]) or f"Evaluated as {base_eval['rating']} lead",
        "triggers": base_eval["triggers"],
        "ai_qualified": False,
    }
