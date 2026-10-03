"""
Triage node — classifies the customer's intent and routes to the right agent.

Reuses the existing classifier from classifier/intent.py rather than
rebuilding classification logic. This node simply bridges the classifier
output into the graph state.
"""

import time

from google.genai.errors import ServerError, ClientError
from classifier.intent import classify_intent, IntentResult
from logger import get_logger

log = get_logger(__name__)

_RETRIES = 2
_BACKOFF_SECONDS = 1.0


def _classify_with_retry(message: str) -> IntentResult:
    """Retry 5xx errors from the classifier before giving up."""
    for attempt in range(_RETRIES + 1):
        try:
            return classify_intent(message)
        except ServerError as e:
            if attempt == _RETRIES:
                raise
            log.warning("Classifier 5xx (attempt %d/%d): %s", attempt + 1, _RETRIES + 1, e)
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))


def triage(state) -> dict:
    """
    Classify the customer's message and populate intent fields in state.

    Falls back to 'faq' intent if the classifier is unavailable (rate
    limited, server error, etc.) — same graceful degradation as before.
    """
    # Get the latest user message
    user_message = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            user_message = msg.content
            break

    if not user_message:
        log.warning("Triage node received empty message list")
        return {"intent": "faq", "confidence": 0.0, "entities": {}}

    try:
        result = _classify_with_retry(user_message)
        log.info(
            "Triage classified: %s (%.0f%%) entities=%s",
            result.intent, result.confidence * 100, result.entities,
        )
        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "entities": result.entities,
        }

    except (ServerError, ClientError) as e:
        log.error("Triage classifier unavailable: %s", e)
        return {"intent": "faq", "confidence": 0.0, "entities": {}}

    except Exception as e:
        log.error("Triage classifier failed: %s", e, exc_info=True)
        return {"intent": "faq", "confidence": 0.0, "entities": {}}
