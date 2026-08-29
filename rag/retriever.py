"""
ChromaDB-based retriever for semantic search over embedded documents.

Handles persistent storage so documents only need to be embedded once.
"""

import chromadb
from rag.embedder import get_embedding

# Persistent ChromaDB client — data survives between runs
_chroma_client = chromadb.PersistentClient(path="./vectorstore")

COLLECTION_NAME = "smarttech_docs"


def get_collection():
    """Get or create the ChromaDB collection."""
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the vector store for chunks most relevant to the query.

    Returns a list of dicts, each containing:
        - text: the chunk content
        - source: the original document filename
        - score: similarity distance (lower = more similar)
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    # Unpack ChromaDB's nested list format into clean dicts
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "score": results["distances"][0][i],
        })

    return chunks
