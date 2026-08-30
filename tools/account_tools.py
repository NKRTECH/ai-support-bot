"""
Account-related tools for the support agent.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "smarttech.db")


def _get_conn():
    return sqlite3.connect(DB_PATH)


def get_customer_info(email: str) -> str:
    """
    Look up a customer's account details by email.

    Args:
        email: The customer's email address.

    Returns:
        A formatted string with customer info, or a not-found message.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT id, name, email, phone, tier, city, created_at FROM customers WHERE email = ?",
        (email.lower(),),
    ).fetchone()
    conn.close()

    if not row:
        return f"No account found for email '{email}'."

    cid, name, email, phone, tier, city, created = row
    return (
        f"Customer ID: {cid}\n"
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Phone: {phone}\n"
        f"Membership Tier: {tier.title()}\n"
        f"City: {city}\n"
        f"Member since: {created}"
    )


def reset_password(email: str) -> str:
    """
    Initiate a password reset for the given email.

    In a real system this would trigger an email. Here it simulates
    the confirmation message.

    Args:
        email: The customer's email address.

    Returns:
        A confirmation message.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT name FROM customers WHERE email = ?", (email.lower(),)
    ).fetchone()
    conn.close()

    if not row:
        return f"No account found for email '{email}'. Cannot send reset link."

    return (
        f"Password reset link sent to {email}. "
        f"The link is valid for 24 hours. "
        f"If {row[0]} doesn't receive it within 5 minutes, "
        f"ask them to check their spam folder."
    )
