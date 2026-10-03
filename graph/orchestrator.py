"""
LangGraph orchestrator — defines the multi-agent support graph.

Graph structure:
    START → triage → [faq | action | escalation] → quality → END
                                                      ↑    ↓
                                                      └── retry (if score < 0.7, max 1)
                                  escalation → END (skip quality)

Conditional routing is based on the intent field set by the triage node.
"""

from langgraph.graph import StateGraph, START, END
from graph.state import SupportState
from graph.nodes.triage import triage
from graph.nodes.faq import faq
from graph.nodes.action import action
from graph.nodes.escalation import escalation
from graph.nodes.quality import quality, QUALITY_THRESHOLD, MAX_RETRIES
from logger import get_logger

log = get_logger(__name__)


# ── Routing functions ────────────────────────────────────────────────────

def route_by_intent(state) -> str:
    """Route from triage to the appropriate agent based on classified intent."""
    intent = state.intent or "faq"

    if intent in ("faq", "technical_issue"):
        return "faq"
    elif intent in ("order_status", "refund_request", "password_reset"):
        return "action"
    elif intent == "escalation":
        return "escalation"
    else:
        # Unknown intent — default to faq
        log.warning("Unknown intent '%s', routing to faq", intent)
        return "faq"


def route_after_quality(state) -> str:
    """
    Route from quality check: accept, retry, or finalize.

    If quality score >= threshold → accept and set final_response
    If score < threshold AND retries left → route back to the original agent
    If score < threshold AND no retries → accept anyway (don't block the customer)
    """
    score = state.quality_score
    retries = state.retry_count
    intent = state.intent or "faq"

    if score >= QUALITY_THRESHOLD:
        log.info("Quality passed (%.2f >= %.2f)", score, QUALITY_THRESHOLD)
        return "accept"

    if retries < MAX_RETRIES:
        log.info(
            "Quality failed (%.2f < %.2f), retry %d/%d",
            score, QUALITY_THRESHOLD, retries + 1, MAX_RETRIES,
        )
        # Route back to the original agent for regeneration
        if intent in ("order_status", "refund_request", "password_reset"):
            return "retry_action"
        return "retry_faq"

    log.info("Quality failed (%.2f) but max retries reached, accepting", score)
    return "accept"


def finalize(state) -> dict:
    """Set final_response from draft_response when quality check passes."""
    return {"final_response": state.draft_response or ""}


def increment_retry(state) -> dict:
    """Bump the retry counter before re-running a node."""
    return {"retry_count": state.retry_count + 1}


# ── Build the graph ──────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the multi-agent support graph."""
    graph = StateGraph(SupportState)

    # Add nodes
    graph.add_node("triage", triage)
    graph.add_node("faq", faq)
    graph.add_node("action", action)
    graph.add_node("escalation", escalation)
    graph.add_node("quality", quality)
    graph.add_node("finalize", finalize)
    graph.add_node("increment_retry", increment_retry)

    # START → triage
    graph.add_edge(START, "triage")

    # triage → [faq | action | escalation]
    graph.add_conditional_edges("triage", route_by_intent, {
        "faq": "faq",
        "action": "action",
        "escalation": "escalation",
    })

    # faq → quality
    graph.add_edge("faq", "quality")

    # action → quality
    graph.add_edge("action", "quality")

    # escalation → END (skip quality check)
    graph.add_edge("escalation", END)

    # quality → [accept | retry_faq | retry_action]
    graph.add_conditional_edges("quality", route_after_quality, {
        "accept": "finalize",
        "retry_faq": "increment_retry",
        "retry_action": "increment_retry",
    })

    # finalize → END
    graph.add_edge("finalize", END)

    # increment_retry → back to the agent
    # We need to route based on intent after incrementing retry
    graph.add_conditional_edges("increment_retry", route_by_intent, {
        "faq": "faq",
        "action": "action",
        "escalation": "escalation",
    })

    return graph.compile()


# Compiled graph instance — import this from app.py
support_graph = build_graph()
