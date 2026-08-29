"""
Recursive text chunker for splitting documents into manageable pieces.

Uses a cascading split strategy: first tries paragraph breaks, then line breaks,
then sentence endings, then falls back to spaces. This preserves semantic
coherence within each chunk as much as possible.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks using recursive character splitting.

    Args:
        text: The full document text to split.
        chunk_size: Target maximum characters per chunk.
        overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try splitting on these separators in order of preference
    separators = ["\n\n", "\n", ". ", " "]

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            return _merge_parts(parts, sep, chunk_size, overlap)

    # If nothing works (e.g., a single giant word), hard-cut
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _merge_parts(
    parts: list[str], sep: str, chunk_size: int, overlap: int
) -> list[str]:
    """
    Merge split parts back into chunks that are under chunk_size,
    then add overlap between consecutive chunks.
    """
    chunks = []
    current = ""

    for part in parts:
        # If adding this part would exceed the limit, save current and start new
        candidate = current + sep + part if current else part
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            # Start new chunk with overlap from end of previous chunk
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + sep + part
            else:
                current = part
        else:
            current = candidate

    # Don't forget the last chunk
    if current.strip():
        chunks.append(current.strip())

    return chunks
