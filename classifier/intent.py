"""
Intent classification using Gemini structured output.

Classifies each incoming customer message into one of several intent
categories and extracts relevant entities (order IDs, error codes, etc.)
so the system can route to the correct handler.
"""

import os
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from logger import get_logger

log = get_logger(__name__)

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_MODEL = "gemini-3.6-flash"



class IntentResult(BaseModel):
    """Structured output schema for intent classification."""

    intent: str = Field(
        description=(
            "The classified intent. Must be one of: "
            "faq, order_status, refund_request, password_reset, "
            "technical_issue, escalation"
        )
    )
    confidence: float = Field(default=0.9)
    order_id: str | None = Field(default=None)
    email: str | None = Field(default=None)
    error_code: str | None = Field(default=None)
    product_name: str | None = Field(default=None)

    @property
    def entities(self) -> dict:
        """Return non-null entities as a dict."""
        out = {}
        if self.order_id:
            out["order_id"] = self.order_id
        if self.email:
            out["email"] = self.email
        if self.error_code:
            out["error_code"] = self.error_code
        if self.product_name:
            out["product_name"] = self.product_name
        return out


CLASSIFICATION_PROMPT = """\
You are an intent classifier for SmartTech customer support.
Classify the customer message into exactly ONE of these intents:

- faq: General questions about policies, products, features, pricing, company info
- order_status: Customer wants to know where their order is or track a shipment
- refund_request: Customer wants a refund or return
- password_reset: Customer needs to reset password or recover their account
- technical_issue: Customer has a technical problem, error code, or needs troubleshooting
- escalation: Customer is angry, wants to speak to a manager, or the issue is beyond support

Respond with this exact JSON structure:
{
  "intent": "<one of the intents above>",
  "confidence": <0.0 to 1.0>,
  "order_id": "<if mentioned, e.g. ORD-1234, else null>",
  "email": "<if mentioned, else null>",
  "error_code": "<if mentioned, e.g. E-401, else null>",
  "product_name": "<if mentioned, e.g. ProBook 15, else null>"
}
"""


def classify_intent(message: str) -> IntentResult:
    """
    Classify a customer message into an intent category.

    Uses a lightweight Gemini call with JSON output to get a valid,
    parseable result every time.
    """
    import json

    response = _client.models.generate_content(
        model=_MODEL,
        contents=f"Customer message: {message}",
        config=types.GenerateContentConfig(
            system_instruction=CLASSIFICATION_PROMPT,
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
        ),
    )

    # Extract text from parts directly (avoids 'non-text parts' warning)
    raw = ""
    if response.candidates and response.candidates[0].content.parts:
        for p in response.candidates[0].content.parts:
            if not getattr(p, 'thought', False) and getattr(p, 'text', None):
                raw += p.text

    data = json.loads(raw)
    log.debug("Intent raw JSON: %s", raw[:300])

    # Handle case where LLM returns entities as a nested dict
    if "entities" in data and isinstance(data["entities"], dict):
        entities = data.pop("entities")
        log.debug("Flattening nested entities dict: %s", entities)
        for key in ("order_id", "email", "error_code", "product_name"):
            if key in entities and entities[key]:
                data.setdefault(key, entities[key])

    result = IntentResult(**data)
    log.debug("Parsed IntentResult: intent=%s confidence=%.2f", result.intent, result.confidence)
    return result

