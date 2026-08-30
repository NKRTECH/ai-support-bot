"""
MCP server for the knowledge base (RAG search).

Exposes hybrid search as a tool, document listing as a resource,
and a support-response prompt template — so any MCP client can
search SmartTech's documentation and format professional replies.

Run standalone:
    python -m mcp_servers.knowledge_server
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer
from rag.retriever import hybrid_search
from rag.reranker import rerank

DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "documents",
)

mcp = MCPServer("smarttech-knowledge")


# ── Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def search_docs(query: str) -> str:
    """
    Search SmartTech's knowledge base using hybrid search (vector + BM25)
    with LLM reranking. Returns the top 5 most relevant document chunks.
    """
    raw_results = hybrid_search(query, top_k=15)
    ranked_results = rerank(query, raw_results, top_n=5)

    if not ranked_results:
        return "No relevant documents found for this query."

    output = []
    for chunk in ranked_results:
        source = chunk.get("source", "unknown")
        score = chunk.get("relevance_score", "?")
        text = chunk["text"]
        output.append(f"[Source: {source}] (relevance: {score})\n{text}")

    return "\n\n---\n\n".join(output)


# ── Resources ────────────────────────────────────────────────────────────

@mcp.resource("docs://list")
def document_list() -> str:
    """List of all knowledge base documents available for search."""
    docs = []
    if os.path.isdir(DOCS_DIR):
        for filename in sorted(os.listdir(DOCS_DIR)):
            if filename.endswith(".md"):
                title = (
                    filename.replace(".md", "")
                    .split("-", 1)[-1]
                    .replace("-", " ")
                    .title()
                )
                docs.append({"filename": filename, "title": title})

    return json.dumps(docs, indent=2)


# ── Prompts ──────────────────────────────────────────────────────────────

@mcp.prompt()
def support_response(question: str, context: str) -> str:
    """
    Generate a professional SmartTech support response given a customer
    question and retrieved context from the knowledge base.
    """
    return (
        "You are a friendly and professional customer support agent "
        "for SmartTech, an Indian consumer electronics brand.\n\n"
        "Use the following knowledge base context to answer the "
        "customer's question. Be concise, use ₹ for prices, and "
        "cite which document the information comes from.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Customer question: {question}"
    )


if __name__ == "__main__":
    mcp.run()
