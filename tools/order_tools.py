"""
Order-related tools for the support agent.

These functions query the SQLite database and return formatted strings
that the LLM can use to answer customer questions about orders.
"""

import sqlite3
import os
from logger import get_logger

log = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "smarttech.db")


def _get_conn():
    return sqlite3.connect(DB_PATH)


def check_order_status(order_id: str) -> str:
    """
    Look up the current status of a specific order.

    Args:
        order_id: The order ID to look up (e.g., "ORD-1015").

    Returns:
        A formatted string with the order details, or an error message
        if the order is not found.
    """
    conn = _get_conn()
    row = conn.execute(
        """
        SELECT o.id, o.product_name, o.price, o.status, o.order_date,
               o.shipping_date, o.delivery_date, o.tracking_number, o.carrier,
               c.name, c.email
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.id = ?
        """,
        (order_id.upper(),),
    ).fetchone()
    conn.close()

    log.debug("check_order_status('%s') -> row=%s", order_id, row is not None)

    if not row:
        return f"No order found with ID '{order_id}'. Please double-check the order number."

    (oid, product, price, status, order_date, ship_date,
     delivery_date, tracking, carrier, cust_name, cust_email) = row

    lines = [
        f"Order: {oid}",
        f"Customer: {cust_name} ({cust_email})",
        f"Product: {product}",
        f"Amount: Rs.{price:,.0f}",
        f"Status: {status.upper()}",
        f"Ordered on: {order_date}",
    ]
    if ship_date:
        lines.append(f"Shipped on: {ship_date} via {carrier}")
    if tracking:
        lines.append(f"Tracking: {tracking}")
    if delivery_date:
        lines.append(f"Delivered on: {delivery_date}")

    return "\n".join(lines)


def list_recent_orders(email: str) -> str:
    """
    List the 5 most recent orders for a customer, looked up by email.

    Args:
        email: The customer's email address.

    Returns:
        A formatted string listing recent orders, or a message if no
        orders are found.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT o.id, o.product_name, o.price, o.status, o.order_date
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE c.email = ?
        ORDER BY o.order_date DESC
        LIMIT 5
        """,
        (email.lower(),),
    ).fetchall()
    conn.close()

    if not rows:
        return f"No orders found for email '{email}'."

    lines = [f"Recent orders for {email}:"]
    for oid, product, price, status, date in rows:
        lines.append(f"  {oid} | {product} | Rs.{price:,.0f} | {status} | {date}")

    return "\n".join(lines)
