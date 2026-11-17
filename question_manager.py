"""WhatsApp AI Assistant Question Manager.

Provides database CRUD, caching, default seed data, and dynamic prompt formatting
for the questions asked by the WhatsApp AI assistant during property enquiries.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn.error")

DEFAULT_WHATSAPP_QUESTIONS = [
    {
        "id": "q_location_preference",
        "slot_key": "location_preference",
        "title": "Location Preference",
        "question_text": "Which area or city are you interested in?",
        "options": [
            "Haldwani (Rampur Rd, Kaladhungi Rd, Bithoria, Mukhani)",
            "Nainital / Bhimtal",
            "Almora",
            "Other (please type the area)",
        ],
        "is_required": True,
        "order": 1,
        "ai_guidance": "When the customer sends any city or location name, give a warm, insightful suggestion praising that city, present at least 3-4 prime residential localities as options, and confirm their choice before advancing.",
        "enabled": True,
    },
    {
        "id": "q_planning_to_buy",
        "slot_key": "planning_to_buy",
        "title": "Buying Timeline",
        "question_text": "When are you planning to buy?",
        "options": [
            "Immediately",
            "Within 1 Month",
            "Within 3 Months",
            "Within 6 Months",
            "Within 12 Months",
            "More Than 1 Year",
            "Just Exploring / Researching",
        ],
        "is_required": True,
        "order": 2,
        "ai_guidance": "Identify purchase urgency to prioritize consultation scheduling.",
        "enabled": True,
    },
    {
        "id": "q_budget_range",
        "slot_key": "budget_range",
        "title": "Budget Range",
        "question_text": "What is your budget range?",
        "options": [
            "Up to 25 lakh",
            "25 lakh - 50 lakh",
            "50 lakh - 1 crore",
            "1 crore - 2 crore",
            "Above 2 crore",
        ],
        "is_required": True,
        "order": 3,
        "ai_guidance": "Reassure them that Profilent has prime verified inventory matching their price point.",
        "enabled": True,
    },
    {
        "id": "q_payment_option",
        "slot_key": "payment_option",
        "title": "Payment Structure",
        "question_text": "How do you plan to pay?",
        "options": [
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
        "is_required": True,
        "order": 4,
        "ai_guidance": "Understand whether buyer is self-funding with cash or requires home loan assistance.",
        "enabled": True,
    },
    {
        "id": "q_bhk_preference",
        "slot_key": "bhk_preference",
        "title": "BHK / Property Configuration",
        "question_text": "Which configuration / BHK do you need?",
        "options": [
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
        "is_required": True,
        "order": 5,
        "ai_guidance": "Validate their configuration choice (e.g. 2 BHK is great for rental yield, 3 BHK for families, Villa for luxury living).",
        "enabled": True,
    },
    {
        "id": "q_remarks_requirements",
        "slot_key": "remarks_requirements",
        "title": "Specific Requirements / Remarks",
        "question_text": "Do you have any specific remarks or requirements (such as parking, floor preference, facing direction, hill view, or nearby schools)? Type whatever you need, or type 'skip' if none.",
        "options": [],
        "is_required": False,
        "order": 6,
        "ai_guidance": "Acknowledge any custom preferences and assure that they will be prioritized.",
        "enabled": True,
    },
]

# In-memory cached active questions
_cached_active_questions: list[dict] = []
_cached_all_questions: list[dict] = []
_last_questions_cache_time = 0.0
_CACHE_TTL = 10.0  # seconds


def invalidate_questions_cache() -> None:
    """Invalidate in-memory cache so subsequent reads query fresh data from MongoDB."""
    global _last_questions_cache_time
    _last_questions_cache_time = 0.0


def _get_sync_db():
    """Get a synchronous MongoDB client/database for thread-pool callers."""
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME", "property_db")
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        return client[db_name]
    except Exception as e:
        logger.warning("Could not connect sync MongoClient for questions: %s", e)
        return None


def serialize_question(doc: dict) -> dict:
    """Serialize a MongoDB question document to a clean JSON-ready dict."""
    if not doc:
        return {}
    q = dict(doc)
    if "_id" in q:
        q["_id"] = str(q["_id"])
    if not q.get("id"):
        q["id"] = q.get("_id") or f"q_{q.get('slot_key', 'custom')}"
    if isinstance(q.get("created_at"), datetime):
        q["created_at"] = q["created_at"].isoformat()
    if isinstance(q.get("updated_at"), datetime):
        q["updated_at"] = q["updated_at"].isoformat()
    return q


def get_active_questions_sync(force_reload: bool = False) -> list[dict]:
    """Retrieve all enabled questions sorted by order synchronously (for Gemini prompt builder)."""
    global _cached_active_questions, _last_questions_cache_time
    now = time.monotonic()
    if not force_reload and _cached_active_questions and (now - _last_questions_cache_time < _CACHE_TTL):
        return _cached_active_questions

    db = _get_sync_db()
    if db is None:
        return DEFAULT_WHATSAPP_QUESTIONS

    try:
        coll = db["whatsapp_questions"]
        count = coll.count_documents({})
        if count == 0:
            # Seed default questions
            now_dt = datetime.now(timezone.utc)
            seed_docs = []
            for item in DEFAULT_WHATSAPP_QUESTIONS:
                doc = dict(item)
                doc["created_at"] = now_dt
                doc["updated_at"] = now_dt
                seed_docs.append(doc)
            coll.insert_many(seed_docs)
            logger.info("Initialized %d default WhatsApp enquiry questions in MongoDB.", len(seed_docs))

        docs = list(coll.find({"enabled": {"$ne": False}}).sort("order", 1))
        serialized = [serialize_question(d) for d in docs]
        _cached_active_questions = serialized or DEFAULT_WHATSAPP_QUESTIONS
        _last_questions_cache_time = now
        return _cached_active_questions
    except Exception as e:
        logger.warning("Failed to fetch active questions from DB; using defaults: %s", e)
        return DEFAULT_WHATSAPP_QUESTIONS


async def get_all_questions_async(app_state) -> list[dict]:
    """Retrieve all questions (enabled and disabled) sorted by order asynchronously."""
    coll = app_state.whatsapp_questions
    count = await coll.count_documents({})
    if count == 0:
        now_dt = datetime.now(timezone.utc)
        seed_docs = []
        for item in DEFAULT_WHATSAPP_QUESTIONS:
            doc = dict(item)
            doc["created_at"] = now_dt
            doc["updated_at"] = now_dt
            seed_docs.append(doc)
        await coll.insert_many(seed_docs)
        logger.info("Initialized %d default WhatsApp enquiry questions asynchronously.", len(seed_docs))

    docs = await coll.find({}).sort("order", 1).to_list(100)
    return [serialize_question(d) for d in docs]


async def get_active_questions_async(app_state) -> list[dict]:
    """Retrieve all enabled questions sorted by order asynchronously."""
    coll = app_state.whatsapp_questions
    count = await coll.count_documents({})
    if count == 0:
        await get_all_questions_async(app_state)
    docs = await coll.find({"enabled": {"$ne": False}}).sort("order", 1).to_list(100)
    return [serialize_question(d) for d in docs]


def format_question_prompt_for_slot(q: dict, lang: str = "en") -> str:
    """Format the English prompt text with numbered options for a question (English only)."""
    q_text = q.get("question_text", "").strip()
    options = q.get("options", [])
    if options:
        opt_lines = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        return f"{q_text}\n{opt_lines}"
    return q_text
