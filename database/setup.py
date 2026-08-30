"""
Database setup script.

Creates a SQLite database with sample customer, order, and refund data
that mirrors what a real SmartTech support system might look like.
Run this once before starting the app.
"""

import sqlite3
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "smarttech.db")


def create_tables(conn):
    """Create the schema."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS customers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            tier TEXT DEFAULT 'standard',
            city TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS orders (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            product_sku TEXT NOT NULL,
            product_name TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            order_date TEXT NOT NULL,
            shipping_date TEXT,
            delivery_date TEXT,
            tracking_number TEXT,
            carrier TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE IF NOT EXISTS refunds (
            id TEXT PRIMARY KEY,
            order_id TEXT NOT NULL,
            amount REAL NOT NULL,
            reason TEXT,
            status TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            processed_at TEXT,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        );
    """)


def seed_data(conn):
    """Populate with realistic sample data."""

    # --- Customers ---
    customers = [
        ("CUST-1001", "Aarav Sharma", "aarav.sharma@gmail.com", "+91-98765-11001", "gold", "Mumbai"),
        ("CUST-1002", "Priya Patel", "priya.patel@gmail.com", "+91-98765-11002", "standard", "Ahmedabad"),
        ("CUST-1003", "Rohan Gupta", "rohan.gupta@yahoo.com", "+91-98765-11003", "platinum", "Delhi"),
        ("CUST-1004", "Sneha Reddy", "sneha.reddy@gmail.com", "+91-98765-11004", "gold", "Hyderabad"),
        ("CUST-1005", "Vikram Singh", "vikram.singh@outlook.com", "+91-98765-11005", "standard", "Jaipur"),
        ("CUST-1006", "Ananya Iyer", "ananya.iyer@gmail.com", "+91-98765-11006", "standard", "Chennai"),
        ("CUST-1007", "Karthik Nair", "karthik.nair@gmail.com", "+91-98765-11007", "gold", "Kochi"),
        ("CUST-1008", "Divya Joshi", "divya.joshi@gmail.com", "+91-98765-11008", "standard", "Pune"),
        ("CUST-1009", "Arjun Menon", "arjun.menon@yahoo.com", "+91-98765-11009", "platinum", "Bangalore"),
        ("CUST-1010", "Meera Krishnan", "meera.k@gmail.com", "+91-98765-11010", "standard", "Coimbatore"),
        ("CUST-1011", "Rahul Verma", "rahul.verma@gmail.com", "+91-98765-11011", "gold", "Lucknow"),
        ("CUST-1012", "Pooja Deshmukh", "pooja.d@outlook.com", "+91-98765-11012", "standard", "Nagpur"),
        ("CUST-1013", "Aditya Rao", "aditya.rao@gmail.com", "+91-98765-11013", "standard", "Mysore"),
        ("CUST-1014", "Ishita Banerjee", "ishita.b@gmail.com", "+91-98765-11014", "gold", "Kolkata"),
        ("CUST-1015", "Nikhil Chopra", "nikhil.chopra@gmail.com", "+91-98765-11015", "standard", "Chandigarh"),
        ("CUST-1016", "Kavya Srinivasan", "kavya.s@yahoo.com", "+91-98765-11016", "platinum", "Bangalore"),
        ("CUST-1017", "Siddharth Malhotra", "sid.malhotra@gmail.com", "+91-98765-11017", "standard", "Delhi"),
        ("CUST-1018", "Tanvi Kulkarni", "tanvi.k@gmail.com", "+91-98765-11018", "standard", "Pune"),
        ("CUST-1019", "Harsh Pandey", "harsh.pandey@gmail.com", "+91-98765-11019", "gold", "Varanasi"),
        ("CUST-1020", "Riya Saxena", "riya.saxena@outlook.com", "+91-98765-11020", "standard", "Indore"),
    ]

    # Assign creation dates (random dates in the past year)
    base_date = datetime(2025, 8, 1)
    for cust in customers:
        created = base_date + timedelta(days=random.randint(0, 365))
        conn.execute(
            "INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)",
            (*cust, created.strftime("%Y-%m-%d")),
        )

    # --- Products (matching the catalog docs) ---
    products = [
        ("ST-PB15-2026", "ProBook 15", 84999),
        ("ST-AB13-2026", "AirBook 13", 57999),
        ("ST-GS16-2026", "GameStation 16", 119999),
        ("ST-ACC-MPAD", "SmartTech Pro Mouse Pad", 999),
        ("ST-ACC-USBCHDMI", "USB-C to HDMI Adapter", 1499),
        ("ST-ACC-KBMECH", "SmartTech MechKeys 75", 4999),
        ("ST-ACC-MOUSE", "SmartTech ErgoMouse", 1999),
        ("ST-ACC-STAND", "SmartTech LaptopStand Pro", 3499),
    ]

    carriers = ["Delhivery", "BlueDart", "DTDC", "Ekart", "Shadowfax"]
    statuses = ["delivered", "delivered", "delivered", "shipped", "processing", "delivered"]

    # --- Orders ---
    order_id = 1001
    for i in range(50):
        cust = random.choice(customers)
        prod = random.choice(products)
        status = random.choice(statuses)
        order_date = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 240))
        shipping_date = None
        delivery_date = None
        tracking = None
        carrier = None

        if status in ("shipped", "delivered"):
            shipping_date = order_date + timedelta(days=random.randint(1, 3))
            carrier = random.choice(carriers)
            tracking = f"{carrier[:3].upper()}{random.randint(100000, 999999)}"
        if status == "delivered":
            delivery_date = shipping_date + timedelta(days=random.randint(2, 7)) if shipping_date else None

        conn.execute(
            "INSERT OR IGNORE INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"ORD-{order_id}",
                cust[0],
                prod[0],
                prod[1],
                prod[2],
                status,
                order_date.strftime("%Y-%m-%d"),
                shipping_date.strftime("%Y-%m-%d") if shipping_date else None,
                delivery_date.strftime("%Y-%m-%d") if delivery_date else None,
                tracking,
                carrier,
            ),
        )
        order_id += 1

    # --- Refunds ---
    # Pick some delivered orders and create refund records
    delivered = conn.execute(
        "SELECT id, price FROM orders WHERE status = 'delivered' LIMIT 10"
    ).fetchall()

    reasons = [
        "Product arrived damaged",
        "Wrong item received",
        "Changed my mind",
        "Found a better price elsewhere",
        "Product not as described",
    ]

    for i, (oid, price) in enumerate(delivered[:10]):
        req_date = datetime(2026, 7, 1) + timedelta(days=random.randint(0, 60))
        status = random.choice(["approved", "approved", "pending", "rejected"])
        processed = None
        if status in ("approved", "rejected"):
            processed = req_date + timedelta(days=random.randint(1, 5))

        conn.execute(
            "INSERT OR IGNORE INTO refunds VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"REF-{2001 + i}",
                oid,
                price,
                random.choice(reasons),
                status,
                req_date.strftime("%Y-%m-%d"),
                processed.strftime("%Y-%m-%d") if processed else None,
            ),
        )

    conn.commit()


def setup():
    """Create the database and populate it."""
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)
    seed_data(conn)

    # Print summary
    counts = {
        "customers": conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
        "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
        "refunds": conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0],
    }
    conn.close()

    print(f"Database created at: {DB_PATH}")
    for table, count in counts.items():
        print(f"  {table}: {count} records")


if __name__ == "__main__":
    setup()
