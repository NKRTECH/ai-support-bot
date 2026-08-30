"""
Intent classification using Gemini structured output.

Classifies each incoming customer message into one of several intent
categories and extracts relevant entities (order IDs, error codes, etc.)
so the system can route to the correct handler.
"""

import json
import os
from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from logger import get_logger

log = get_logger(__name__)

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# _MODEL = "gemini-3.6-flash"
_MODEL = "gemma-4-31b-it"


class IntentResult(BaseModel):
    """Structured output schema for intent classification."""

    intent: Literal[
        "faq", "order_status", "refund_request",
        "password_reset", "technical_issue", "escalation",
    ] = Field(
        description="The classified intent category."
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
"""


def classify_intent(message: str) -> IntentResult:
    """
    Classify a customer message into an intent category.

    Uses Gemini structured output with response_schema to guarantee
    valid, schema-compliant JSON. The Literal type on intent constrains
    the model to only output one of the 6 valid intent categories.
    """
    response = _client.models.generate_content(
        model=_MODEL,
        contents=f"Customer message: {message}",
        config=types.GenerateContentConfig(
            system_instruction=CLASSIFICATION_PROMPT,
            temperature=0.0,
            max_output_tokens=256,
            response_mime_type="application/json",
            response_schema=IntentResult,
        ),
    )

    # Extract text from parts (skip thinking tokens)
    raw = ""
    if response.candidates and response.candidates[0].content.parts:
        for p in response.candidates[0].content.parts:
            if not getattr(p, 'thought', False) and getattr(p, 'text', None):
                raw += p.text

    log.debug("Intent raw JSON: %s", raw[:300])

    # Extract JSON object if model returned surrounding text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]

    data = json.loads(raw)

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
