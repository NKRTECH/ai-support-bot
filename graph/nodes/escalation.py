"""
Escalation node — summarizes the conversation and hands off to a human agent.

Generates an empathetic response and provides contact information.
Skips quality check since escalation responses are formulaic.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from graph.nodes.utils import extract_text
from logger import get_logger

load_dotenv()
log = get_logger(__name__)


def escalation(state) -> dict:
    """
    Summarize the conversation and route to human support.

    Sets final_response directly (bypasses quality check) since
    escalation responses follow a fixed pattern.
    """
    # Get the latest user message
    user_message = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            user_message = msg.content
            break

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.5,
        max_output_tokens=512,
    )

    prompt = (
        "The customer wants to escalate to a human agent. "
        "Be empathetic, acknowledge their frustration, and tell them "
        "you're connecting them with a senior support specialist. "
        "Provide the toll-free number 1800-123-4567 and WhatsApp "
        "+91-98765-43210 for immediate assistance.\n\n"
        f"Customer message: {user_message}"
    )

    response = llm.invoke([
        SystemMessage(content=(
            "You are a friendly SmartTech customer support agent. "
            "Be empathetic and professional."
        )),
        HumanMessage(content=prompt),
    ])

    text = extract_text(response.content)
    log.info("Escalation response generated: %d chars", len(text))

    # Set final_response directly — skip quality check
    return {"final_response": text}
