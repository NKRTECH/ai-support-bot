"""
SmartTech Customer Support Agent
Multi-agent system orchestrated by LangGraph with RAG, tool calling,
and quality-check reflection loop.
"""

import logging
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from logger import get_logger
from agent.guardrails import validate_input
from graph.orchestrator import support_graph

log = get_logger(__name__)

# google-genai prints an AFC warning on the first call; we don't use AFC
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def main():
    """Main chat loop — reads user input, runs the LangGraph agent, prints response."""

    # Validate API key
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
        print("=" * 60)
        print("ERROR: No Gemini API key found!")
        print()
        print("1. Get a free key at: https://aistudio.google.com")
        print("2. Create a .env file in this folder with:")
        print("   GEMINI_API_KEY=your_actual_key_here")
        print("=" * 60)
        return

    # Welcome message
    print("=" * 60)
    print("  SmartTech Customer Support")
    print("  How can I help you today?")
    print("=" * 60)
    print()
    print("Type your message and press Enter.")
    print('Type "quit" or "exit" to end the conversation.')
    print()

    # Chat loop
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Thanks for contacting SmartTech support.")
            break

        # Skip empty messages
        if not user_input:
            continue

        # Validate input
        is_valid, reason = validate_input(user_input)
        if not is_valid:
            print(f"\nBot: {reason}\n")
            continue

        # Exit commands
        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\nBot: Goodbye! Thanks for contacting SmartTech support. "
                  "Have a great day!")
            break

        # Run the multi-agent graph
        try:
            result = support_graph.invoke({
                "messages": [HumanMessage(content=user_input)],
            })

            # Extract the final response
            response = result.get("final_response", "")
            if not response:
                response = result.get("draft_response", "")
            if not response:
                response = ("I wasn't able to generate a response. "
                            "Please try again or contact us at 1800-123-4567.")

            # Show routing info
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            quality_score = result.get("quality_score", 0.0)
            retry_count = result.get("retry_count", 0)

            status_parts = [f"{intent} ({confidence:.0%})"]
            if quality_score > 0:
                status_parts.append(f"quality: {quality_score:.2f}")
            if retry_count > 0:
                status_parts.append(f"retries: {retry_count}")

            print(f"\033[90m[{' | '.join(status_parts)}]\033[0m")
            print(f"\nBot: {response.strip()}\n")

        except Exception as e:
            log.error("Graph execution failed: %s", e, exc_info=True)
            print("\nBot: I ran into an issue processing your request. "
                  "Please try again or call us at 1800-123-4567.\n")


if __name__ == "__main__":
    main()
