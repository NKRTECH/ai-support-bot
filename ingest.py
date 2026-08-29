"""
Document ingestion script.

Reads all markdown files from the documents/ folder, chunks them,
generates embeddings via Gemini, and stores everything in a persistent
ChromaDB collection. Safe to re-run — skips documents that are already ingested.
"""

import os
import glob
from rag.chunker import chunk_text
from rag.embedder import get_embeddings
from rag.retriever import get_collection

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")


def ingest():
    """Read, chunk, embed, and store all documents."""
    collection = get_collection()

    # Find all markdown files
    doc_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))

    if not doc_paths:
        print(f"No .md files found in {DOCS_DIR}")
        return

    print(f"Found {len(doc_paths)} documents to process.\n")

    total_chunks = 0
    skipped_docs = 0

    for doc_path in doc_paths:
        filename = os.path.basename(doc_path)

        # Check if this document was already ingested (by checking metadata)
        existing = collection.get(where={"source": filename})
        if existing and existing["ids"]:
            print(f"  [skip] {filename} -- already ingested ({len(existing['ids'])} chunks), skipping.")
            skipped_docs += 1
            total_chunks += len(existing["ids"])
            continue

        # Read the file
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Chunk it
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        if not chunks:
            print(f"  [warn] {filename} -- empty after chunking, skipping.")
            continue

        print(f"  [+] {filename} -- {len(chunks)} chunks, embedding...", end=" ", flush=True)

        # Generate embeddings for all chunks in this document
        embeddings = get_embeddings(chunks, delay=0.3)

        # Prepare IDs and metadata for ChromaDB
        ids = [f"{filename}__chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # Upsert into the collection
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks += len(chunks)
        print("done.")

    print(f"\nDone! {total_chunks} total chunks in the vector store.")
    if skipped_docs:
        print(f"({skipped_docs} documents were already ingested and skipped.)")


if __name__ == "__main__":
    ingest()
