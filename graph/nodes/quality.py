"""
Quality check node, implements the Reflection pattern.

Reviews the draft response from the faq/action nodes and scores it on
accuracy, tone and completeness. If the score is below the threshold the
graph routes back for one retry.

The judge uses response_schema so the model has to return valid JSON.
Asking for "only JSON" in the prompt was not reliable enough.
"""

import os
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from llm_utils import response_text, parse_json_object
from logger import get_logger

load_dotenv()
log = get_logger(__name__)

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_MODEL = "gemini-3.8-flash"

QUALITY_THRESHOLD = 0.7
MAX_RETRIES = 1
_SCORING_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0
_UNSCORED_FALLBACK = 0.8  # if the judge is down, let the draft through


class QualityScore(BaseModel):
    """Structured output schema for quality scoring."""
    accuracy: float = Field(description="Is the information correct and grounded? (0.0-1.0)")
    tone: float = Field(description="Is it professional and helpful? (0.0-1.0)")
    completeness: float = Field(description="Does it fully answer the question? (0.0-1.0)")
    overall: float = Field(description="Average of the three scores (0.0-1.0)")


SCORING_INSTRUCTIONS = (
    "You are a quality reviewer for a customer support bot. "
    "Score the draft response on three criteria, each from 0.0 to 1.0: "
    "accuracy (correct information?), tone (professional and helpful?), "
    "completeness (fully answers the question?). "
    "Set overall to the average of the three scores."
)


def _score_draft(user_message: str, draft: str) -> QualityScore:
    """One judge call. Raises on API errors, empty output or bad JSON."""
    response = _client.models.generate_content(
        model=_MODEL,
        contents=f"Customer question: {user_message}\n\nDraft response: {draft}",
        config=types.GenerateContentConfig(
            system_instruction=SCORING_INSTRUCTIONS,
            temperature=0.0,
            # thinking tokens count against this limit
            max_output_tokens=2048,
            response_mime_type="application/json",
            response_schema=QualityScore,
        ),
    )
    raw = response_text(response)
    if not raw.strip():
        finish = response.candidates[0].finish_reason if response.candidates else None
        raise ValueError(f"empty judge response (finish_reason={finish})")
    return QualityScore(**parse_json_object(raw))


def quality(state) -> dict:
    """
    Score the draft response and decide whether to retry or accept.

    If the judge fails twice the draft is accepted anyway, and the error is
    logged so a broken judge doesn't go unnoticed.
    """
    draft = state.draft_response
    if not draft:
        log.warning("Quality check received empty draft")
        return {
            "quality_score": 0.0,
            "final_response": "I wasn't able to generate a response. "
                              "Please try again or contact us at 1800-123-4567.",
        }

    # Get the user's original question
    user_message = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            user_message = msg.content
            break

    for attempt in range(1, _SCORING_ATTEMPTS + 1):
        try:
            scores = _score_draft(user_message, draft)
        except Exception as e:
            log.warning("Quality judge attempt %d/%d failed: %s", attempt, _SCORING_ATTEMPTS, e)
            if attempt < _SCORING_ATTEMPTS:
                time.sleep(_BACKOFF_SECONDS * attempt)
            continue

        overall = max(0.0, min(1.0, scores.overall))
        log.info(
            "Quality scores: accuracy=%.2f tone=%.2f completeness=%.2f overall=%.2f",
            scores.accuracy, scores.tone, scores.completeness, overall,
        )
        return {"quality_score": overall, "retry_count": state.retry_count}

    log.error(
        "Quality UNSCORED after %d attempts, accepting draft with fallback %.2f",
        _SCORING_ATTEMPTS, _UNSCORED_FALLBACK,
    )
    return {"quality_score": _UNSCORED_FALLBACK, "retry_count": state.retry_count}
