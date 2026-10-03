"""
Helpers for pulling JSON out of raw google-genai responses.

gemma-4-31b-it returns reasoning parts (thought=True) before the answer, and
sometimes wraps the JSON in ```json fences even when JSON output is requested.
"""

import json


def response_text(response) -> str:
    """Join the answer text of a google-genai response, skipping thought parts."""
    if not response.candidates or not response.candidates[0].content.parts:
        return ""
    return "".join(
        p.text
        for p in response.candidates[0].content.parts
        if getattr(p, "text", None) and not getattr(p, "thought", False)
    )


def parse_json_object(raw: str) -> dict:
    """
    Parse the JSON object in raw model output.

    Slices from the first '{' to the last '}' so fences and stray text are
    ignored. Raises ValueError (JSONDecodeError is a subclass) on failure.
    """
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model output: {raw[:120]!r}")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data
