"""
Embedding wrapper for Google Gemini's embedding model.

Provides single and batch embedding functions with built-in rate limiting
to stay within the free tier's request-per-minute quota.
"""

import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Gemini client (shared across calls)
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-2"


def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for a single text string."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def get_embeddings(texts: list[str], delay: float = 0.3) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Adds a small delay between API calls to avoid hitting
    the free tier rate limit (roughly 1500 RPM for embeddings).
    """
    embeddings = []
    for i, text in enumerate(texts):
        embeddings.append(get_embedding(text))
        if i < len(texts) - 1:
            time.sleep(delay)
    return embeddings
