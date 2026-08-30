"""
Hybrid retriever combining semantic (vector) search with BM25 keyword search.

Vector search excels at understanding meaning ("laptop for college" matches
"student notebook"). BM25 excels at exact matches ("E-401" matches "E-401").
Combining both and re-ranking gives the best of both worlds.
"""

import os
import chromadb
from rank_bm25 import BM25Okapi
from rag.embedder import get_embedding

# Persistent ChromaDB client — absolute path so it works regardless of cwd
_VECTORSTORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vectorstore",
)
_chroma_client = chromadb.PersistentClient(path=_VECTORSTORE_PATH)

COLLECTION_NAME = "smarttech_docs"

# BM25 index — built lazily on first search
_bm25_index = None
_bm25_chunks = None


def get_collection():
    """Get or create the ChromaDB collection."""
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _build_bm25_index():
    """Build the BM25 index from all chunks in ChromaDB (called once)."""
    global _bm25_index, _bm25_chunks

    collection = get_collection()
    all_data = collection.get(include=["documents", "metadatas"])

    if not all_data["documents"]:
        _bm25_chunks = []
        _bm25_index = None
        return

    _bm25_chunks = []
    tokenized = []

    for i, doc in enumerate(all_data["documents"]):
        _bm25_chunks.append({
            "id": all_data["ids"][i],
            "text": doc,
            "source": all_data["metadatas"][i].get("source", "unknown"),
        })
        # Tokenize by lowercasing and splitting on whitespace
        tokenized.append(doc.lower().split())

    _bm25_index = BM25Okapi(tokenized)


def _vector_search(query: str, top_k: int = 20) -> list[dict]:
    """Semantic search via ChromaDB embeddings."""
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score": results["distances"][0][i],
            "method": "vector",
        })
    return chunks


def _bm25_search(query: str, top_k: int = 20) -> list[dict]:
    """Keyword search via BM25."""
    global _bm25_index, _bm25_chunks

    if _bm25_index is None:
        _build_bm25_index()

    if not _bm25_chunks or _bm25_index is None:
        return []

    tokenized_query = query.lower().split()
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-k indices sorted by score (descending)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    chunks = []
    for idx, score in ranked:
        if score > 0:
            chunk = _bm25_chunks[idx].copy()
            chunk["score"] = float(score)
            chunk["method"] = "bm25"
            chunks.append(chunk)

    return chunks


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Basic vector-only search (kept for backward compatibility).
    Use hybrid_search() for better results.
    """
    return _vector_search(query, top_k)


def hybrid_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Combine vector and BM25 search results, deduplicate, and return
    a merged list. Re-ranking is done separately via rag.reranker.
    """
    vector_results = _vector_search(query, top_k=20)
    bm25_results = _bm25_search(query, top_k=20)

    # Merge and deduplicate by chunk ID
    seen_ids = set()
    merged = []

    for chunk in vector_results + bm25_results:
        chunk_id = chunk.get("id", chunk["text"][:50])
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            merged.append(chunk)

    # If no reranker is applied, just return the first top_k
    # (vector results come first, so they're prioritized)
    return merged[:top_k]
