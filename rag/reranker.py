"""
Lightweight LLM-based reranker.

Takes a query and a list of candidate chunks, asks the LLM to score
each chunk's relevance on a 0-10 scale, and returns the top-N results
sorted by relevance score.
"""

import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_MODEL = "gemini-3.6-flash"

RERANK_PROMPT = """\
You are a relevance scorer. Given a search query and a text chunk,
rate how relevant the chunk is to answering the query.

Score from 0 to 10:
- 0: Completely irrelevant
- 5: Somewhat related but doesn't directly answer
- 10: Directly and completely answers the query

Respond with ONLY a JSON object: {"score": <number>}
"""


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-rank a list of retrieved chunks by LLM-judged relevance.

    Each chunk dict must have at least a 'text' key.
    Returns the top_n chunks sorted by relevance (highest first),
    with a 'relevance_score' field added.
    """
    scored = []

    for chunk in chunks:
        try:
            response = _client.models.generate_content(
                model=_MODEL,
                contents=(
                    f"Query: {query}\n\n"
                    f"Text chunk: {chunk['text'][:500]}"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=RERANK_PROMPT,
                    temperature=0.0,
                    max_output_tokens=32,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(response.text)
            chunk["relevance_score"] = float(data.get("score", 0))
        except Exception:
            # If scoring fails for a chunk, give it a neutral score
            chunk["relevance_score"] = 5.0

        scored.append(chunk)

    # Sort by relevance (highest first) and return top N
    scored.sort(key=lambda c: c["relevance_score"], reverse=True)
    return scored[:top_n]
