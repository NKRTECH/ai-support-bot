"""
FAQ node — handles knowledge base questions using RAG.

Runs hybrid search + reranking on the knowledge base, then generates
a grounded answer using the retrieved context. Reuses the existing
RAG pipeline from rag/retriever.py and rag/reranker.py.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from rag.retriever import hybrid_search
from rag.reranker import rerank
from graph.nodes.utils import extract_text
from logger import get_logger

load_dotenv()
log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a friendly and professional customer support agent for SmartTech, "
    "an Indian consumer electronics brand. Use the provided knowledge base "
    "context to answer the customer's question. Be concise, use INR for "
    "prices, and cite which document the information comes from. If the "
    "context doesn't contain the answer, say so honestly."
)


def faq(state) -> dict:
    """
    Search the knowledge base and generate a grounded answer.

    Pipeline: hybrid_search → rerank → LLM with context injection.
    """
    # Get the latest user message
    user_message = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            user_message = msg.content
            break

    if not user_message:
        return {"draft_response": "I couldn't find your question. Could you rephrase that?"}

    # RAG pipeline
    raw_results = hybrid_search(user_message, top_k=15)
    log.debug("FAQ hybrid search: %d raw results", len(raw_results))

    ranked_results = rerank(user_message, raw_results, top_n=5)
    log.debug("FAQ reranked to %d results", len(ranked_results))

    # Build context block
    if ranked_results:
        context_block = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in ranked_results
        )
        retrieved_context = context_block
    else:
        context_block = ""
        retrieved_context = ""

    # Generate answer with context
    prompt = (
        f"--- KNOWLEDGE BASE CONTEXT ---\n{context_block}\n"
        f"--- END CONTEXT ---\n\n"
        f"Customer question: {user_message}"
    ) if context_block else user_message

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.7,
        max_output_tokens=1024,
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    text = extract_text(response.content)
    log.info("FAQ generated response: %d chars", len(text))
    return {
        "draft_response": text,
        "retrieved_context": retrieved_context,
    }
