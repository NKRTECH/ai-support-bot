"""
MCP server for customer account operations.

Exposes account tools (get customer info, reset password) and an
accounts://tiers resource describing SmartTech's loyalty tiers.

Run standalone:
    python -m mcp_servers.accounts_server
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer
from tools.account_tools import get_customer_info, reset_password

mcp = MCPServer("smarttech-accounts")


# ── Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def get_customer_info_tool(email: str) -> str:
    """Look up a customer's account details by email address."""
    return get_customer_info(email)


@mcp.tool()
def reset_password_tool(email: str) -> str:
    """Send a password reset link to the customer's email address."""
    return reset_password(email)


# ── Resources ────────────────────────────────────────────────────────────

TIER_DESCRIPTIONS = """\
SmartTech Loyalty Tiers:

1. Bronze (default)
   - Earned automatically on signup
   - Standard shipping rates
   - Email support

2. Silver (₹25,000+ lifetime spend)
   - Free standard shipping on all orders
   - Priority email support (24h response)
   - 5% discount on accessories

3. Gold (₹75,000+ lifetime spend)
   - Free express shipping on all orders
   - Priority phone + chat support
   - 10% discount on accessories
   - Early access to new product launches

4. Platinum (₹1,50,000+ lifetime spend)
   - Free same-day delivery (select cities)
   - Dedicated account manager
   - 15% discount on all products
   - Extended 30-day return window
   - Invitation to SmartTech annual event"""


@mcp.resource("accounts://tiers")
def loyalty_tiers() -> str:
    """SmartTech's loyalty program tiers and their benefits."""
    return TIER_DESCRIPTIONS


if __name__ == "__main__":
    mcp.run()
