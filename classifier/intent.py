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
    confidence: float = Field(
        description="Confidence score from 0.0 to 1.0"
    )
    entities: dict = Field(
        default_factory=dict,
        description=(
            "Extracted entities like order_id, email, error_code, "
            "product_name. Empty dict if none found."
        ),
    )


CLASSIFICATION_PROMPT = """\
You are an intent classifier for SmartTech customer support.
Classify the customer message into exactly ONE of these intents:

- faq: General questions about policies, products, features, pricing, company info
- order_status: Customer wants to know where their order is or track a shipment
- refund_request: Customer wants a refund or return
- password_reset: Customer needs to reset password or recover their account
- technical_issue: Customer has a technical problem, error code, or needs troubleshooting
- escalation: Customer is angry, wants to speak to a manager, or the issue is beyond support

Also extract any relevant entities from the message:
- order_id: e.g., "ORD-1234"
- email: e.g., "user@example.com"
- error_code: e.g., "E-401"
- product_name: e.g., "ProBook 15"

Respond with JSON matching the schema exactly.
"""


def classify_intent(message: str) -> IntentResult:
    """
    Classify a customer message into an intent category.

    Uses a lightweight Gemini call with structured output (JSON schema)
    to ensure we always get a valid, parseable result.
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

    # Parse the structured JSON response into our Pydantic model
    import json

    data = json.loads(response.text)
    return IntentResult(**data)
