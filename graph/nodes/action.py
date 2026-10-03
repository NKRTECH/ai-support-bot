"""
Action node — handles order, refund, and account operations using tool calling.

Uses LangChain's tool abstraction with ChatGoogleGenerativeAI for the ReAct
loop. Wraps existing tool functions as LangChain tools. Preserves HITL
guardrails for high-risk operations (refunds).
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.order_tools import (
    check_order_status as _check_order_status,
    list_recent_orders as _list_recent_orders,
)
from tools.account_tools import (
    get_customer_info as _get_customer_info,
    reset_password as _reset_password,
)
from tools.refund_tools import (
    check_refund_eligibility as _check_refund_eligibility,
    process_refund as _process_refund,
)
from agent.guardrails import needs_approval, request_approval
from graph.nodes.utils import extract_text
from logger import get_logger

load_dotenv()
log = get_logger(__name__)

MAX_TOOL_STEPS = 10

SYSTEM_PROMPT = (
    "You are a customer support agent for SmartTech. You have access to tools "
    "for checking orders, processing refunds, and managing accounts. "
    "Use the tools to find information and help the customer. "
    "Be concise and professional. Use INR for prices."
)


# ── LangChain tool wrappers ─────────────────────────────────────────────
# These wrap the existing Python functions so LangChain can auto-generate
# schemas and pass them to the model via bind_tools().

@tool
def check_order_status(order_id: str) -> str:
    """Look up the current status of a specific order by order ID (e.g. ORD-1015)."""
    return _check_order_status(order_id)


@tool
def list_recent_orders(email: str) -> str:
    """List the 5 most recent orders for a customer by their email address."""
    return _list_recent_orders(email)


@tool
def get_customer_info(email: str) -> str:
    """Look up a customer's account details by email address."""
    return _get_customer_info(email)


@tool
def reset_password(email: str) -> str:
    """Send a password reset link to the customer's email address."""
    return _reset_password(email)


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check whether an order is eligible for a refund based on delivery date and return policy."""
    return _check_refund_eligibility(order_id)


@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund for a delivered order. Requires order ID and reason."""
    return _process_refund(order_id, reason)


ALL_TOOLS = [
    check_order_status,
    list_recent_orders,
    get_customer_info,
    reset_password,
    check_refund_eligibility,
    process_refund,
]

TOOL_MAP = {t.name: t for t in ALL_TOOLS}

# Tools that require human approval before execution
APPROVAL_REQUIRED = {"process_refund"}


def action(state) -> dict:
    """
    Run a ReAct tool-calling loop to handle action intents.

    The LLM can chain multiple tool calls (e.g., check order → check
    eligibility → process refund). Includes HITL approval for refunds.
    """
    # Get the latest user message
    user_message = ""
    for msg in reversed(state.messages):
        if msg.type == "human":
            user_message = msg.content
            break

    if not user_message:
        return {"draft_response": "I couldn't understand your request. Could you try again?"}

    llm = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        temperature=0.3,
        max_output_tokens=1024,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Build the conversation for the ReAct loop
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    tool_results = []

    for step in range(MAX_TOOL_STEPS):
        log.info("Action agent step %d/%d", step + 1, MAX_TOOL_STEPS)

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if the model wants to call tools
        if not response.tool_calls:
            # No more tool calls — the model is done
            log.info("Action agent finished at step %d", step + 1)
            return {
                "draft_response": extract_text(response.content) or "",
                "tool_results": tool_results,
            }

        # Process each tool call
        from langchain_core.messages import ToolMessage

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            log.info("Action tool call [step %d]: %s(%s)", step + 1, tool_name, tool_args)
            print(f"\033[90m[calling: {tool_name}({tool_args})]\033[0m")

            # HITL guardrail for refunds
            if tool_name in APPROVAL_REQUIRED:
                print(f"\033[90m[requires human approval]\033[0m")
                approved = request_approval(tool_name, tool_args)
                if not approved:
                    result = (
                        f"Action DENIED by supervisor. The {tool_name} request was "
                        f"not approved. Inform the customer politely and offer "
                        f"alternatives (e.g., contacting support directly)."
                    )
                    log.info("Tool call denied by operator: %s", tool_name)
                    print("\033[90m[action denied by operator]\033[0m")
                else:
                    result = TOOL_MAP[tool_name].invoke(tool_args)
                    print("\033[90m[tool result received]\033[0m")
            elif tool_name in TOOL_MAP:
                result = TOOL_MAP[tool_name].invoke(tool_args)
                print("\033[90m[tool result received]\033[0m")
            else:
                result = f"Error: Unknown tool '{tool_name}'"
                log.warning("Unknown tool requested: %s", tool_name)

            tool_results.append(f"{tool_name}: {str(result)[:200]}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    # Exhausted steps
    log.warning("Action agent exhausted %d steps", MAX_TOOL_STEPS)
    return {
        "draft_response": (
            "I went through several steps but wasn't able to fully resolve this. "
            "Please contact our support team at 1800-123-4567 for direct help."
        ),
        "tool_results": tool_results,
    }
