"""
Refund-related tools for the support agent.
"""

import sqlite3
import os
from datetime import datetime
from logger import get_logger

log = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "smarttech.db")


def _get_conn():
    return sqlite3.connect(DB_PATH)


def check_refund_eligibility(order_id: str) -> str:
    """
    Check whether an order is eligible for a refund.

    Eligibility rules:
    - Order must exist and be in 'delivered' status
    - Delivery must be within the 15-day return window
    - No existing approved refund for this order

    Args:
        order_id: The order ID to check (e.g., "ORD-1015").

    Returns:
        A string describing whether the order is eligible and why.
    """
    conn = _get_conn()
    order = conn.execute(
        """
        SELECT o.id, o.product_name, o.price, o.status, o.delivery_date, c.name
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.id = ?
        """,
        (order_id.upper(),),
    ).fetchone()

    if not order:
        conn.close()
        return f"No order found with ID '{order_id}'."

    oid, product, price, status, delivery_date, cust_name = order

    # Check if already refunded
    existing = conn.execute(
        "SELECT status FROM refunds WHERE order_id = ? AND status = 'approved'",
        (oid,),
    ).fetchone()
    conn.close()

    if existing:
        return f"Order {oid} ({product}) has already been refunded."

    if status != "delivered":
        return (
            f"Order {oid} is currently '{status}'. "
            f"Refunds can only be processed for delivered orders."
        )

    if not delivery_date:
        return f"Order {oid} has no delivery date recorded. Please contact support."

    # Check 15-day window
    delivered = datetime.strptime(delivery_date, "%Y-%m-%d")
    days_since = (datetime.now() - delivered).days

    if days_since > 15:
        return (
            f"Order {oid} ({product}) was delivered {days_since} days ago. "
            f"Unfortunately, our 15-day return window has passed. "
            f"For warranty claims, please contact support directly."
        )

    return (
        f"Order {oid} is ELIGIBLE for a refund.\n"
        f"Customer: {cust_name}\n"
        f"Product: {product}\n"
        f"Refund amount: Rs.{price:,.0f}\n"
        f"Delivered: {delivery_date} ({days_since} days ago)\n"
        f"Within the 15-day return window."
    )


def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund for the given order.

    Creates a refund record in the database with 'approved' status.

    Args:
        order_id: The order ID to refund (e.g., "ORD-1015").
        reason: The reason for the refund.

    Returns:
        A confirmation message with refund details.
    """
    conn = _get_conn()
    order = conn.execute(
        "SELECT id, product_name, price FROM orders WHERE id = ?",
        (order_id.upper(),),
    ).fetchone()

    if not order:
        conn.close()
        return f"No order found with ID '{order_id}'. Cannot process refund."

    oid, product, price = order

    # Check for existing refund
    existing = conn.execute(
        "SELECT id FROM refunds WHERE order_id = ? AND status = 'approved'",
        (oid,),
    ).fetchone()

    if existing:
        conn.close()
        return f"Order {oid} already has an approved refund ({existing[0]})."

    # Create the refund
    now = datetime.now().strftime("%Y-%m-%d")
    refund_id = f"REF-{3001 + hash(oid) % 1000}"

    conn.execute(
        "INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?, ?)",
        (refund_id, oid, price, reason, "approved", now, now),
    )
    conn.commit()
    conn.close()

    return (
        f"Refund {refund_id} has been processed.\n"
        f"Order: {oid} ({product})\n"
        f"Refund amount: Rs.{price:,.0f}\n"
        f"Reason: {reason}\n"
        f"Status: APPROVED\n"
        f"The refund will be credited to the original payment method "
        f"within 5-7 business days."
    )
