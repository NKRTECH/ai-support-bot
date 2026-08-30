"""
MCP server for order and refund operations.

Exposes SmartTech order tools (check status, list orders, refund
eligibility, process refund) and an orders://recent resource via
the Model Context Protocol.

Run standalone:
    python -m mcp_servers.orders_server
"""

import sys
import os
import json
import sqlite3

# Add project root to path so tool imports work when run standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.mcpserver import MCPServer
from tools.order_tools import check_order_status, list_recent_orders
from tools.refund_tools import check_refund_eligibility, process_refund

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "database", "smarttech.db",
)

mcp = MCPServer("smarttech-orders")


# ── Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def check_order_status_tool(order_id: str) -> str:
    """Look up the current status of a specific order by order ID (e.g. ORD-1015)."""
    return check_order_status(order_id)


@mcp.tool()
def list_recent_orders_tool(email: str) -> str:
    """List the 5 most recent orders for a customer by their email address."""
    return list_recent_orders(email)


@mcp.tool()
def check_refund_eligibility_tool(order_id: str) -> str:
    """Check whether an order is eligible for a refund based on delivery date and return policy."""
    return check_refund_eligibility(order_id)


@mcp.tool()
def process_refund_tool(order_id: str, reason: str) -> str:
    """Process a refund for a delivered order. Requires order ID and reason."""
    return process_refund(order_id, reason)


# ── Resources ────────────────────────────────────────────────────────────

@mcp.resource("orders://recent")
def recent_orders() -> str:
    """The 10 most recent orders across all customers."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT o.id, o.product_name, o.price, o.status, o.order_date,
               c.name, c.email
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        ORDER BY o.order_date DESC
        LIMIT 10
        """
    ).fetchall()
    conn.close()

    orders = [
        {
            "order_id": r[0], "product": r[1], "price": r[2],
            "status": r[3], "date": r[4], "customer": r[5], "email": r[6],
        }
        for r in rows
    ]

    return json.dumps(orders, indent=2)


if __name__ == "__main__":
    mcp.run()
