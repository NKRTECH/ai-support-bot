"""
SmartTech Customer Support Agent
RAG-powered conversational interface using the Google Gemini API
and a ChromaDB knowledge base.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError
from logger import get_logger
from classifier.intent import classify_intent, IntentResult
from rag.retriever import hybrid_search
from rag.reranker import rerank
from tools.order_tools import check_order_status, list_recent_orders
from tools.account_tools import get_customer_info, reset_password
from tools.refund_tools import check_refund_eligibility, process_refund
from agent.loop import run_agent
from agent.guardrails import validate_input

log = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemma-4-31b-it"
# MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are a friendly and professional customer support agent for SmartTech, \
an Indian consumer electronics brand headquartered in Bangalore. SmartTech sells laptops, \
peripherals, and accessories across India.

Your role:
- Help customers with orders, returns, shipping, technical issues, and product questions
- Be professional but warm — like a helpful friend who works at the company
- Use ₹ (INR) when mentioning any prices
- Be concise — customers want quick answers, not essays
- If you don't know something specific (like an order status or exact policy detail), \
say so honestly. Don't make up information.
- When answering from provided context, mention which document/policy the info comes from.
- If the context does not contain the answer, say "I don't have that information in our \
knowledge base" and suggest contacting support directly.

SmartTech quick facts:
- Website: smarttech.in
- Support phone: 1800-123-4567 (toll-free)
- WhatsApp: +91-98765-43210
- Support hours: Mon-Sat 9AM-9PM IST (chat is 24/7)
- 15-day return policy on all products
- 2-year warranty on laptops
- Free shipping on orders over ₹1,999
"""


def main():
    """Main chat loop — reads user input, sends to Gemini, prints response."""

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

    # Initialize the Gemini client
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Create a chat session with system instructions
    # The chat session automatically maintains conversation history
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        ),
    )

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
            print("\n\nGoodbye! Thanks for contacting SmartTech support. 👋")
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
                  "Have a great day! 👋")
            break

        # Classify the customer's intent
        classification_failed = False
        try:
            intent_result = classify_intent(user_input)
            log.info(
                "Intent classified: %s (%.0f%%) entities=%s | input='%s'",
                intent_result.intent, intent_result.confidence * 100,
                intent_result.entities, user_input[:100],
            )
        except ServerError as e:
            log.error("Intent classification unavailable: %s", e)
            intent_result = IntentResult(intent="faq")
            classification_failed = True
            print("\033[90m[classifier unavailable, falling back to faq]\033[0m")
        except ClientError as e:
            log.error("Intent classification client error: %s", e)
            intent_result = IntentResult(intent="faq")
            classification_failed = True
            if e.code == 429:
                print("\033[90m[rate limited, falling back to faq]\033[0m")
            else:
                print("\033[90m[classifier error, falling back to faq]\033[0m")
        except Exception as e:
            log.error("Intent classification failed: %s", e, exc_info=True)
            intent_result = IntentResult(intent="faq")
            classification_failed = True
            print("\033[90m[classifier error, falling back to faq]\033[0m")

        # Show the classified intent
        entities_str = ""
        if intent_result.entities:
            entities_str = f" | entities: {intent_result.entities}"
        print(f"\033[90m[{intent_result.intent} ({intent_result.confidence:.0%}){entities_str}]\033[0m")

        # Step 2: Route based on intent
        if intent_result.intent in ("faq", "technical_issue"):
            _handle_faq(chat, user_input)

        elif intent_result.intent in ("order_status", "refund_request", "password_reset"):
            _handle_action(client, user_input, intent_result)

        elif intent_result.intent == "escalation":
            _handle_escalation(chat, user_input)

        else:
            _handle_faq(chat, user_input)


def _handle_faq(chat, user_input):
    """Hybrid search + rerank + streamed RAG answer."""
    raw_results = hybrid_search(user_input, top_k=15)
    log.debug("Hybrid search returned %d raw results for: '%s'", len(raw_results), user_input[:80])
    ranked_results = rerank(user_input, raw_results, top_n=5)
    log.debug(
        "Reranked to %d results. Top scores: %s",
        len(ranked_results),
        [f"{c.get('relevance_score', '?')}" for c in ranked_results[:3]],
    )

    if ranked_results:
        context_block = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in ranked_results
        )
        augmented_input = (
            f"--- KNOWLEDGE BASE CONTEXT ---\n{context_block}\n"
            f"--- END CONTEXT ---\n\n"
            f"Customer question: {user_input}"
        )
    else:
        augmented_input = user_input

    _stream_response(chat, augmented_input)


def _handle_escalation(chat, user_input):
    """Empathetic handoff to human support."""
    augmented_input = (
        f"The customer wants to escalate. Be empathetic, acknowledge their "
        f"frustration, and tell them you're connecting them with a senior "
        f"support specialist. Provide the toll-free number 1800-123-4567 "
        f"and WhatsApp +91-98765-43210 for immediate assistance."
        f"\n\nCustomer message: {user_input}"
    )
    _stream_response(chat, augmented_input)


# ── Tool declarations + registry (shared between handler and agent loop) ────

TOOL_REGISTRY = {
    "check_order_status": check_order_status,
    "list_recent_orders": list_recent_orders,
    "get_customer_info": get_customer_info,
    "reset_password": reset_password,
    "check_refund_eligibility": check_refund_eligibility,
    "process_refund": process_refund,
}

TOOL_DECLARATIONS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="check_order_status",
        description="Look up the current status of a specific order by order ID",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. ORD-1015"}
            },
            "required": ["order_id"],
        },
    ),
    types.FunctionDeclaration(
        name="list_recent_orders",
        description="List the 5 most recent orders for a customer by their email address",
        parameters={
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email address"}
            },
            "required": ["email"],
        },
    ),
    types.FunctionDeclaration(
        name="get_customer_info",
        description="Look up a customer's account details by email",
        parameters={
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email address"}
            },
            "required": ["email"],
        },
    ),
    types.FunctionDeclaration(
        name="reset_password",
        description="Send a password reset link to the customer's email",
        parameters={
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email address"}
            },
            "required": ["email"],
        },
    ),
    types.FunctionDeclaration(
        name="check_refund_eligibility",
        description="Check whether an order is eligible for a refund based on delivery date and return policy",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. ORD-1015"}
            },
            "required": ["order_id"],
        },
    ),
    types.FunctionDeclaration(
        name="process_refund",
        description="Process a refund for a delivered order",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to refund"},
                "reason": {"type": "string", "description": "Reason for the refund"},
            },
            "required": ["order_id", "reason"],
        },
    ),
])


def _handle_action(client, user_input, intent_result):
    """
    Run the ReAct agent loop for action intents.

    The agent can chain multiple tool calls (e.g., check order → check
    eligibility → process refund) and includes human-in-the-loop approval
    for high-risk actions like refunds.
    """
    response_text = run_agent(
        client=client,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_message=user_input,
        tool_declarations=TOOL_DECLARATIONS,
        tool_registry=TOOL_REGISTRY,
    )
    print(f"\nBot: {response_text.strip()}\n")


def _stream_response(chat, message):
    """Stream a chat response token-by-token with thinking display."""
    try:
        thinking_started = False
        response_started = False

        for chunk in chat.send_message_stream(message):
            if not chunk.candidates or not chunk.candidates[0].content.parts:
                continue

            for part in chunk.candidates[0].content.parts:
                if getattr(part, 'thought', False) and getattr(part, 'text', None):
                    if not thinking_started:
                        thinking_started = True
                        col = 0
                        print("\n\033[90m💭 Thinking:\033[0m")
                        print("\033[90m│ \033[0m", end="", flush=True)
                    for char in part.text:
                        if char == '\n':
                            col = 0
                            print(f"\n\033[90m│ \033[0m", end="", flush=True)
                        elif col >= 76:
                            col = 1
                            print(f"\n\033[90m│ \033[0m\033[90m{char}\033[0m", end="", flush=True)
                        else:
                            col += 1
                            print(f"\033[90m{char}\033[0m", end="", flush=True)

                elif getattr(part, 'text', None):
                    if thinking_started and not response_started:
                        print(f"\n\033[90m{'─' * 40}\033[0m")
                    if not response_started:
                        response_started = True
                        print("\nBot: ", end="", flush=True)
                    print(part.text, end="", flush=True)

        if response_started:
            print("\n")

    except ServerError:
        log.error("Streaming hit server error (503/429)", exc_info=True)
        print("\n\nBot: Our service is experiencing high demand right now. "
              "Please try again in a moment.\n")
    except ClientError as e:
        log.error("Streaming client error: %s", e, exc_info=True)
        print("\n\nBot: I ran into an issue processing your request. "
              "Please try again or call us at 1800-123-4567.\n")
    except Exception as e:
        log.error("Streaming failed unexpectedly: %s", e, exc_info=True)
        print("\n\nBot: I ran into an issue processing your request. "
              "Please try again or call us at 1800-123-4567.\n")


if __name__ == "__main__":
    main()
