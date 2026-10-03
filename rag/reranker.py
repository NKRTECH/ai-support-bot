"""
Lightweight LLM-based reranker.

Takes a query and a list of candidate chunks, asks the LLM to score every
chunk's relevance on a 0-10 scale in a single call, and returns the top-N
results sorted by relevance score.

One listwise call replaces the old one-call-per-chunk approach: 15 calls per
question were slow (minutes on gemma) and each one could fail on its own with
a 500/503 or an empty response.
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

_ATTEMPTS = 3
_BACKOFF_SECONDS = 2.0


class RerankScores(BaseModel):
    """Structured output schema: one score per chunk, in input order."""
    scores: list[float] = Field(
        description="Relevance score from 0 to 10 for each chunk, in the same order as the chunks"
    )


RERANK_PROMPT = """\
You are a relevance scorer. You will get a search query and a numbered list
of text chunks. Rate how relevant each chunk is to answering the query.

Score from 0 to 10:
- 0: Completely irrelevant
- 5: Somewhat related but doesn't directly answer
- 10: Directly and completely answers the query

Return exactly one score per chunk, in the same order as the chunks.
"""


def _score_chunks(query: str, chunks: list[dict]) -> list[float]:
    """Score all chunks in one LLM call, retrying on API errors or bad output."""
    numbered = "\n\n".join(
        f"[{i}] {chunk['text'][:500]}" for i, chunk in enumerate(chunks)
    )
    contents = f"Query: {query}\n\nChunks:\n{numbered}"

    for attempt in range(1, _ATTEMPTS + 1):
        try:
            response = _client.models.generate_content(
                model=_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=RERANK_PROMPT,
                    temperature=0.0,
                    max_output_tokens=2048,  # thinking tokens count against this limit
                    response_mime_type="application/json",
                    response_schema=RerankScores,
                ),
            )
            scores = RerankScores(**parse_json_object(response_text(response))).scores
            if len(scores) != len(chunks):
                raise ValueError(f"got {len(scores)} scores for {len(chunks)} chunks")
            return scores
        except Exception as e:
            log.warning("Rerank attempt %d/%d failed: %s", attempt, _ATTEMPTS, e)
            if attempt == _ATTEMPTS:
                raise
            time.sleep(_BACKOFF_SECONDS * attempt)


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-rank a list of retrieved chunks by LLM-judged relevance.

    Each chunk dict must have at least a 'text' key.
    Returns the top_n chunks sorted by relevance (highest first),
    with a 'relevance_score' field added. If scoring fails, the chunks come
    back in their original hybrid-search order without scores.
    """
    if not chunks:
        return []

    try:
        scores = _score_chunks(query, chunks)
    except Exception:
        log.error("Rerank failed, keeping hybrid search order")
        return chunks[:top_n]

    for chunk, score in zip(chunks, scores):
        chunk["relevance_score"] = score

    # sorted() is stable, so ties keep their hybrid-search order
    return sorted(chunks, key=lambda c: c["relevance_score"], reverse=True)[:top_n]
