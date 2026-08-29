"""
SmartTech Customer Support Agent
RAG-powered conversational interface using the Google Gemini API
and a ChromaDB knowledge base.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# MODEL = "gemini-3.1-flash-lite"
MODEL = "gemini-3.6-flash"

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

        # Exit commands
        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\nBot: Goodbye! Thanks for contacting SmartTech support. "
                  "Have a great day! 👋")
            break

        # Step 1: Classify the customer's intent
        from classifier.intent import classify_intent
        from rag.retriever import hybrid_search
        from rag.reranker import rerank

        try:
            intent_result = classify_intent(user_input)
        except Exception as e:
            print(f"\n[intent error: {e}, defaulting to faq]\n")
            from classifier.intent import IntentResult
            intent_result = IntentResult(intent="faq", confidence=0.5, entities={})

        # Show the classified intent
        entities_str = ""
        if intent_result.entities:
            entities_str = f" | entities: {intent_result.entities}"
        print(f"\033[90m[{intent_result.intent} ({intent_result.confidence:.0%}){entities_str}]\033[0m")

        # Step 2: Route based on intent
        if intent_result.intent in ("faq", "technical_issue"):
            # Hybrid search → rerank → build context
            raw_results = hybrid_search(user_input, top_k=15)
            ranked_results = rerank(user_input, raw_results, top_n=5)

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

        elif intent_result.intent == "order_status":
            order_id = intent_result.entities.get("order_id", "unknown")
            augmented_input = (
                f"The customer wants to check order status. "
                f"Order ID: {order_id}. "
                f"We don't have order lookup tools yet. Apologize and tell them "
                f"to call 1800-123-4567 or email support@smarttech.in with their order ID."
                f"\n\nCustomer message: {user_input}"
            )

        elif intent_result.intent == "refund_request":
            order_id = intent_result.entities.get("order_id", "unknown")
            augmented_input = (
                f"The customer wants a refund. Order ID: {order_id}. "
                f"We don't have refund processing tools yet. Walk them through "
                f"the return process from our policy and suggest contacting support."
                f"\n\nCustomer message: {user_input}"
            )

        elif intent_result.intent == "password_reset":
            email = intent_result.entities.get("email", "unknown")
            augmented_input = (
                f"The customer needs a password reset. Email: {email}. "
                f"We don't have account tools yet. Guide them to smarttech.in/account "
                f"and the 'Forgot Password' link, or suggest contacting support."
                f"\n\nCustomer message: {user_input}"
            )

        elif intent_result.intent == "escalation":
            augmented_input = (
                f"The customer wants to escalate. Be empathetic, acknowledge their "
                f"frustration, and tell them you're connecting them with a senior "
                f"support specialist. Provide the toll-free number 1800-123-4567 "
                f"and WhatsApp +91-98765-43210 for immediate assistance."
                f"\n\nCustomer message: {user_input}"
            )

        else:
            # Fallback: treat as FAQ
            augmented_input = user_input

        # Step 3: Stream the LLM response
        try:
            thinking_started = False
            response_started = False

            for chunk in chat.send_message_stream(augmented_input):
                if not chunk.candidates or not chunk.candidates[0].content.parts:
                    continue

                for part in chunk.candidates[0].content.parts:
                    # Stream thinking tokens in grey with word-wrapping
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

                    # Stream response tokens normally
                    elif getattr(part, 'text', None):
                        if thinking_started and not response_started:
                            print(f"\n\033[90m{'─' * 40}\033[0m")
                        if not response_started:
                            response_started = True
                            print("\nBot: ", end="", flush=True)
                        print(part.text, end="", flush=True)

            if response_started:
                print("\n")

        except Exception as e:
            print(f"\n[error] {e}")
            print("Please try again or check your API key.\n")


if __name__ == "__main__":
    main()
