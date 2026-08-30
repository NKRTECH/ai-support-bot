"""
Guardrails for the support agent.

Defines approval gates, input validation, and safety checks
that the agent loop evaluates before executing any tool call.
"""

from logger import get_logger

log = get_logger(__name__)

# Refund threshold — anything above this needs human approval
REFUND_APPROVAL_THRESHOLD = 3000  # ₹3,000


def needs_approval(tool_name: str, arguments: dict) -> bool:
    """
    Check whether a tool call requires human approval before execution.

    Returns True if the action is high-risk (e.g., large refunds,
    account deletions). The agent loop should pause and request
    explicit confirmation before proceeding.
    """
    if tool_name == "process_refund":
        # We don't know the amount here (it's in the DB, not the args),
        # so we always require approval for refund processing.
        # A more sophisticated system would look up the order amount first.
        log.info("Approval required: process_refund for %s", arguments)
        return True

    return False


def request_approval(tool_name: str, arguments: dict) -> bool:
    """
    Prompt the human operator for explicit approval of a high-risk action.

    Blocks until the operator types 'yes' or 'no'. Returns True if approved.
    """
    print()
    print("\033[93m" + "=" * 50 + "\033[0m")
    print("\033[93m⚠️  APPROVAL REQUIRED\033[0m")
    print(f"\033[93m   Action: {tool_name}\033[0m")

    for key, value in arguments.items():
        print(f"\033[93m   {key}: {value}\033[0m")

    print("\033[93m" + "=" * 50 + "\033[0m")

    while True:
        try:
            response = input("\033[93mApprove this action? [yes/no]: \033[0m").strip().lower()
        except (EOFError, KeyboardInterrupt):
            log.info("Approval interrupted — denying")
            return False

        if response in ("yes", "y"):
            log.info("Action approved: %s(%s)", tool_name, arguments)
            return True
        elif response in ("no", "n"):
            log.info("Action denied: %s(%s)", tool_name, arguments)
            return False
        else:
            print("Please type 'yes' or 'no'.")


def validate_input(message: str) -> tuple[bool, str]:
    """
    Basic input validation before processing a customer message.

    Returns (is_valid, reason). If invalid, the reason explains why
    so the caller can inform the customer.
    """
    if not message or not message.strip():
        return False, "Message is empty."

    if len(message) > 2000:
        return False, "Message is too long (max 2000 characters)."

    return True, ""
