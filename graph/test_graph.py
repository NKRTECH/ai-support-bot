"""
Quick smoke test for the LangGraph support agent.

Tests each path: faq, action, and escalation.
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from langchain_core.messages import HumanMessage
from graph.orchestrator import support_graph


def test_case(name: str, message: str):
    """Run a single test case through the graph."""
    print(f"\n{'=' * 60}")
    print(f"  TEST: {name}")
    print(f"  Input: {message}")
    print(f"{'=' * 60}")

    result = support_graph.invoke({
        "messages": [HumanMessage(content=message)],
    })

    intent = result.get("intent", "?")
    confidence = result.get("confidence", 0.0)
    quality_score = result.get("quality_score", 0.0)
    retry_count = result.get("retry_count", 0)
    response = result.get("final_response") or result.get("draft_response", "")

    print(f"\n  Intent: {intent} ({confidence:.0%})")
    print(f"  Quality: {quality_score:.2f}")
    print(f"  Retries: {retry_count}")
    print(f"  Response: {response[:200]}...")
    print(f"  PASS" if response else f"  FAIL")

    return bool(response)


def main():
    results = []

    # Test FAQ path
    results.append(test_case(
        "FAQ - Return Policy",
        "What is your return policy?",
    ))

    # Test action path
    results.append(test_case(
        "Action - Order Status",
        "What's the status of my order ORD-1001?",
    ))

    # Test escalation path
    results.append(test_case(
        "Escalation",
        "I want to speak to a manager right now!",
    ))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} passed")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
