"""Gemini integration for natural-language understanding and response phrasing.

If no API key is configured, callers fall back to deterministic heuristics.
"""
import base64
import json
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("finpilot.gemini")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Free-tier quotas are per-model — if the configured model is rate-limited
# (429) or unavailable, fall back through these in order.
FALLBACK_MODELS = ["gemini-3.1-flash-lite", "gemini-3.6-flash",
                   "gemini-3-flash-preview", "gemini-flash-latest"]


def _candidate_models() -> list[str]:
    return [m for m in [settings.gemini_model] + FALLBACK_MODELS if m]


async def _post(model: str, payload: dict) -> dict | None:
    url = GEMINI_URL.format(model=model)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                url, params={"key": settings.gemini_api_key}, json=payload)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        logger.warning("Gemini call failed on %s: %s", model, e)
        return None


def _extract_text(data: dict) -> str | None:
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def available() -> bool:
    return bool(settings.gemini_api_key)


async def generate(prompt: str, system: str = "", temperature: float = 0.4,
                   json_mode: bool = False) -> str | None:
    """Call Gemini. Returns text or None on failure."""
    if not available():
        return None
    parts = [{"text": prompt}]
    if system:
        parts.insert(0, {"text": f"[System instructions]\n{system}\n\n[User request]\n"})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }
    for model in _candidate_models():
        data = await _post(model, payload)
        if data:
            text = _extract_text(data)
            if text is not None:
                return text
    return None


async def classify_intent(message: str, history: list[dict] | None = None) -> dict | None:
    """Use Gemini to extract intent + slots from a user message."""
    system = (
        "You are the intent parser for FinPilot AI, a personal finance assistant. "
        "Classify the user's message and extract slots. Respond ONLY with JSON: "
        '{"intent": "<one of: spending_query, income_query, why_spending, budget_query, '
        'bill_query, goal_query, add_expense, add_income, add_bill, add_goal, '
        'update_income, affordability, reduce_spending, report, trip_plan, smalltalk, other>", '
        '"amount": <number or null>, "category": <string or null>, '
        '"month": "YYYY-MM" or null, "description": <string or null>, '
        '"goal_name": <string or null>, "target_amount": <number or null>, '
        '"target_months": <number or null>, "needs_confirmation": <bool>, '
        '"confidence": <0-1>}. '
        "needs_confirmation is true when the user asks to create/update/delete financial records."
    )
    ctx = ""
    if history:
        ctx = "Recent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in history[-6:]) + "\n\n"
    out = await generate(ctx + message, system=system, temperature=0.1, json_mode=True)
    if not out:
        return None
    try:
        # strip code fences if present
        txt = out.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").removeprefix("json").strip()
        return json.loads(txt)
    except Exception:
        return None


async def analyze_receipt_image(image_bytes: bytes, mime_type: str) -> str | None:
    """Send a receipt/screenshot to Gemini vision and get raw JSON text back."""
    if not available():
        return None
    prompt = (
        "Extract every expense/transaction visible in this image (a receipt, "
        "bill, payment screenshot, or bank-statement crop). Respond ONLY with a "
        "JSON array of objects, each with keys: "
        '"amount" (number, the money spent), '
        '"date" ("YYYY-MM-DD" if visible, else null), '
        '"description" (merchant/item text), '
        '"category" (one of: Food & Groceries, Transportation, Shopping, '
        'Housing & Rent, Utilities, Healthcare, Education, Entertainment, '
        'Travel, Personal Care, Subscriptions, Other), '
        '"payment_method" (Cash/Card/UPI/NetBanking/Wallet or null). '
        "Rules: ignore subtotal/total/tax/discount lines unless they are actual "
        "purchased items; ignore account balances; if nothing is found return []."
    )
    payload = {
        "contents": [{"role": "user", "parts": [
            {"inline_data": {"mime_type": mime_type,
                             "data": base64.b64encode(image_bytes).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }
    for model in _candidate_models():
        data = await _post(model, payload)
        if data:
            text = _extract_text(data)
            if text is not None:
                return text
    return None


async def phrase_response(instruction: str, facts: dict) -> str | None:
    """Ask Gemini to phrase a factual finance answer using provided facts only."""
    system = (
        "You are FinPilot AI, a concise personal-finance assistant. "
        "Write a short, friendly reply using ONLY the facts provided. "
        "Never invent numbers. Use ₹ for INR. 2-4 sentences plus a suggested next step."
    )
    prompt = f"Facts (JSON):\n{json.dumps(facts, default=str)[:4000]}\n\nInstruction: {instruction}"
    return await generate(prompt, system=system, temperature=0.5)
