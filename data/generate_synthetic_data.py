"""
data/generate_synthetic_data.py
================================
Synthetic data generator for AtlasCare.

Generates realistic JSON data files conforming strictly to the
provided schemas:
  - data/orders.json
  - data/crm_cases.json
  - data/kb_articles.json
  - data/payment_config.json
  - data/sessions.json
  - data/refunds.json  (empty — populated at runtime)

Coverage strategy
-----------------
Data is NOT random. Every record is purposefully designed to cover:

  Refund boundaries
    - ORD with total = Rs.24,999  (just below threshold)
    - ORD with total = Rs.25,000  (exactly at threshold)
    - ORD with total = Rs.25,001  (just above threshold)
    - ORD with total = Rs.42,000  (J3 high-value scenario)

  Order statuses
    - placed, processing, shipped, delivered, cancelled

  Item complexity
    - single-item orders
    - multi-item orders (3 items)
    - orders with a pre-cancelled item

  Customer diversity
    - home address only
    - home + office addresses
    - multiple addresses
    - different loyalty tiers: standard, silver, gold, platinum

  Failure scenarios
    - invalid order (for test injection)
    - already-cancelled item
    - no office address customer (CUST-003)

Usage
-----
  python data/generate_synthetic_data.py
  python data/generate_synthetic_data.py --output-dir ./data
"""

import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _date(days_from_now: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(days=days_from_now)
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

def build_customers() -> list[dict[str, Any]]:
    return [
        {
            "customer_id": "CUST-001",
            "name": "Priya Sharma",
            "email": "priya.sharma@email.com",
            "phone": "+91-9876543210",
            "tier": "gold",
            "order_ids": [
                "ORD-78321", "ORD-78322", "ORD-78323",
                "ORD-78324", "ORD-78325", "ORD-78400",
                "ORD-78401", "ORD-78402", "ORD-78500",
            ],
            "preferred_refund_method": "HDFC_CREDIT",
            "addresses": [
                {
                    "label": "home",
                    "line1": "12 MG Road, Koramangala",
                    "city": "Bengaluru",
                    "state": "Karnataka",
                    "pincode": "560034",
                },
                {
                    "label": "office",
                    "line1": "4th Floor, Prestige Tower, Outer Ring Road",
                    "city": "Bengaluru",
                    "state": "Karnataka",
                    "pincode": "560103",
                },
            ],
        },
        {
            "customer_id": "CUST-002",
            "name": "Arjun Mehta",
            "email": "arjun.mehta@email.com",
            "phone": "+91-9812345678",
            "tier": "platinum",
            "order_ids": ["ORD-99001", "ORD-99002"],
            "preferred_refund_method": "UPI",
            "addresses": [
                {
                    "label": "home",
                    "line1": "Flat 3B, Sunshine Apartments, Bandra West",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "pincode": "400050",
                },
                {
                    "label": "office",
                    "line1": "WeWork, BKC, Bandra Kurla Complex",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "pincode": "400051",
                },
            ],
        },
        {
            # No office address — tests missing address scenario
            "customer_id": "CUST-003",
            "name": "Divya Nair",
            "email": "divya.nair@email.com",
            "phone": "+91-9988776655",
            "tier": "standard",
            "order_ids": ["ORD-88001"],
            "preferred_refund_method": "original",
            "addresses": [
                {
                    "label": "home",
                    "line1": "22 Anna Salai, Teynampet",
                    "city": "Chennai",
                    "state": "Tamil Nadu",
                    "pincode": "600018",
                },
            ],
        },
        {
            "customer_id": "CUST-004",
            "name": "Rahul Gupta",
            "email": "rahul.gupta@email.com",
            "phone": "+91-9001234567",
            "tier": "silver",
            "order_ids": ["ORD-77001", "ORD-77002"],
            "preferred_refund_method": "ICICI_DEBIT",
            "addresses": [
                {
                    "label": "home",
                    "line1": "B-45, Sector 62, Noida",
                    "city": "Noida",
                    "state": "Uttar Pradesh",
                    "pincode": "201301",
                },
                {
                    "label": "office",
                    "line1": "Tower C, Cyber City, DLF Phase 2",
                    "city": "Gurugram",
                    "state": "Haryana",
                    "pincode": "122002",
                },
                {
                    "label": "parents",
                    "line1": "12 Civil Lines",
                    "city": "Allahabad",
                    "state": "Uttar Pradesh",
                    "pincode": "211001",
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def build_orders() -> list[dict[str, Any]]:
    home_blr = {
        "line1": "12 MG Road", "line2": "Koramangala",
        "city": "Bengaluru", "state": "Karnataka", "pincode": "560034",
    }
    home_mum = {
        "line1": "Flat 3B, Sunshine Apartments, Bandra West",
        "city": "Mumbai", "state": "Maharashtra", "pincode": "400050",
    }
    home_chn = {
        "line1": "22 Anna Salai",
        "city": "Chennai", "state": "Tamil Nadu", "pincode": "600018",
    }
    home_noi = {
        "line1": "B-45, Sector 62",
        "city": "Noida", "state": "Uttar Pradesh", "pincode": "201301",
    }

    return [
        # ── CUST-001 ──────────────────────────────────────────────────────
        # J1+J2: multi-item, processing
        {
            "order_id": "ORD-78321", "customer_id": "CUST-001",
            "status": "processing",
            "created_at": _ts(5), "estimated_delivery": _date(2),
            "tracking_number": None,
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-LAPTOP-001",
                 "name": "Dell Inspiron 15 Laptop",
                 "quantity": 1, "unit_price": 55000.00, "status": "active"},
                {"line_id": 2, "product_id": "PROD-BAG-021",
                 "name": "Laptop Backpack",
                 "quantity": 1, "unit_price": 1500.00, "status": "active"},
                {"line_id": 3, "product_id": "PROD-MOUSE-005",
                 "name": "Wireless Mouse",
                 "quantity": 1, "unit_price": 800.00, "status": "active"},
            ],
            "total_amount": 57300.00,
            "payment_method": "HDFC_CREDIT",
        },
        # Shipped — has tracking number (J1 variant)
        {
            "order_id": "ORD-78322", "customer_id": "CUST-001",
            "status": "shipped",
            "created_at": _ts(7), "estimated_delivery": _date(1),
            "tracking_number": "TRACK-7X9K2M",
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-TV-008",
                 "name": "Sony Bravia 55in TV",
                 "quantity": 1, "unit_price": 75000.00, "status": "active"},
            ],
            "total_amount": 75000.00,
            "payment_method": "HDFC_CREDIT",
        },
        # Delivered — immutable for cancellation
        {
            "order_id": "ORD-78323", "customer_id": "CUST-001",
            "status": "delivered",
            "created_at": _ts(14), "estimated_delivery": _date(-7),
            "tracking_number": "TRACK-ABC123",
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-SHOE-042",
                 "name": "Nike Air Max 270",
                 "quantity": 2, "unit_price": 8500.00, "status": "active"},
            ],
            "total_amount": 17000.00,
            "payment_method": "UPI",
        },
        # Placed — eligible for cancellation, no tracking
        {
            "order_id": "ORD-78324", "customer_id": "CUST-001",
            "status": "placed",
            "created_at": _ts(1), "estimated_delivery": _date(5),
            "tracking_number": None,
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-BOOK-101",
                 "name": "Clean Code by Robert Martin",
                 "quantity": 1, "unit_price": 750.00, "status": "active"},
            ],
            "total_amount": 750.00,
            "payment_method": "UPI",
        },
        # Cancelled
        {
            "order_id": "ORD-78325", "customer_id": "CUST-001",
            "status": "cancelled",
            "created_at": _ts(10), "estimated_delivery": _date(-5),
            "tracking_number": None,
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-CHAIR-007",
                 "name": "Ergonomic Office Chair",
                 "quantity": 1, "unit_price": 18000.00, "status": "cancelled"},
            ],
            "total_amount": 0.00,
            "payment_method": "SBI_NETBANKING",
        },
        # Boundary: Rs.24,999 — below threshold
        {
            "order_id": "ORD-78400", "customer_id": "CUST-001",
            "status": "processing",
            "created_at": _ts(2), "estimated_delivery": _date(3),
            "tracking_number": None,
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-WATCH-009",
                 "name": "Fossil Gen 6 Smartwatch",
                 "quantity": 1, "unit_price": 24999.00, "status": "active"},
            ],
            "total_amount": 24999.00,
            "payment_method": "HDFC_CREDIT",
        },
        # Boundary: Rs.25,000 — exactly at threshold
        {
            "order_id": "ORD-78401", "customer_id": "CUST-001",
            "status": "processing",
            "created_at": _ts(2), "estimated_delivery": _date(3),
            "tracking_number": None,
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-TABLET-011",
                 "name": "Samsung Galaxy Tab S8",
                 "quantity": 1, "unit_price": 25000.00, "status": "active"},
            ],
            "total_amount": 25000.00,
            "payment_method": "HDFC_CREDIT",
        },
        # Boundary: Rs.25,001 — above threshold → must escalate
        {
            "order_id": "ORD-78402", "customer_id": "CUST-001",
            "status": "processing",
            "created_at": _ts(2), "estimated_delivery": _date(3),
            "tracking_number": None,
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-PHONE-022",
                 "name": "iPhone 15",
                 "quantity": 1, "unit_price": 25001.00, "status": "active"},
            ],
            "total_amount": 25001.00,
            "payment_method": "ICICI_DEBIT",
        },
        # J3: Rs.42,000 high-value damaged item
        {
            "order_id": "ORD-78500", "customer_id": "CUST-001",
            "status": "delivered",
            "created_at": _ts(20), "estimated_delivery": _date(-12),
            "tracking_number": "TRACK-XY9900",
            "shipping_address": home_blr,
            "items": [
                {"line_id": 1, "product_id": "PROD-LAPTOP-HV",
                 "name": "Apple MacBook Pro 14in",
                 "quantity": 1, "unit_price": 42000.00, "status": "active"},
            ],
            "total_amount": 42000.00,
            "payment_method": "HDFC_CREDIT",
        },
        # ── CUST-002 (cross-customer security tests) ──────────────────────
        {
            "order_id": "ORD-99001", "customer_id": "CUST-002",
            "status": "processing",
            "created_at": _ts(3), "estimated_delivery": _date(4),
            "tracking_number": None,
            "shipping_address": home_mum,
            "items": [
                {"line_id": 1, "product_id": "PROD-HEADPHONE-031",
                 "name": "Sony WH-1000XM5",
                 "quantity": 1, "unit_price": 29000.00, "status": "active"},
            ],
            "total_amount": 29000.00,
            "payment_method": "UPI",
        },
        {
            "order_id": "ORD-99002", "customer_id": "CUST-002",
            "status": "shipped",
            "created_at": _ts(6), "estimated_delivery": _date(1),
            "tracking_number": "TRACK-MUM001",
            "shipping_address": home_mum,
            "items": [
                {"line_id": 1, "product_id": "PROD-PERFUME-011",
                 "name": "Chanel No 5 Perfume",
                 "quantity": 2, "unit_price": 9500.00, "status": "active"},
            ],
            "total_amount": 19000.00,
            "payment_method": "ICICI_DEBIT",
        },
        # ── CUST-003 ──────────────────────────────────────────────────────
        {
            "order_id": "ORD-88001", "customer_id": "CUST-003",
            "status": "delivered",
            "created_at": _ts(30), "estimated_delivery": _date(-20),
            "tracking_number": "TRACK-CHN001",
            "shipping_address": home_chn,
            "items": [
                {"line_id": 1, "product_id": "PROD-SAREE-055",
                 "name": "Kanjeevaram Silk Saree",
                 "quantity": 1, "unit_price": 4500.00, "status": "active"},
            ],
            "total_amount": 4500.00,
            "payment_method": "SBI_NETBANKING",
        },
        # ── CUST-004 ──────────────────────────────────────────────────────
        {
            "order_id": "ORD-77001", "customer_id": "CUST-004",
            "status": "processing",
            "created_at": _ts(2), "estimated_delivery": _date(3),
            "tracking_number": None,
            "shipping_address": home_noi,
            "items": [
                {"line_id": 1, "product_id": "PROD-FRIDGE-002",
                 "name": "LG Double Door Refrigerator",
                 "quantity": 1, "unit_price": 35000.00, "status": "active"},
            ],
            "total_amount": 35000.00,
            "payment_method": "ICICI_DEBIT",
        },
        {
            "order_id": "ORD-77002", "customer_id": "CUST-004",
            "status": "placed",
            "created_at": _ts(0), "estimated_delivery": _date(7),
            "tracking_number": None,
            "shipping_address": home_noi,
            "items": [
                {"line_id": 1, "product_id": "PROD-AC-018",
                 "name": "Daikin 1.5 Ton AC",
                 "quantity": 1, "unit_price": 38000.00, "status": "active"},
                {"line_id": 2, "product_id": "PROD-COVER-009",
                 "name": "AC Cover",
                 "quantity": 1, "unit_price": 500.00, "status": "active"},
            ],
            "total_amount": 38500.00,
            "payment_method": "ICICI_DEBIT",
        },
    ]


# ---------------------------------------------------------------------------
# KB articles
# ---------------------------------------------------------------------------

def build_kb_articles() -> list[dict[str, Any]]:
    return [
        {
            "article_id": "KB-001",
            "title": "Refund Policy and Threshold Rules",
            "tags": ["refund", "threshold", "payments", "escalation", "return"],
            "content": (
                "Acme Retail processes refunds automatically for amounts up to "
                "Rs.25,000 via the Payments Gateway. Refund requests exceeding "
                "Rs.25,000 must be escalated to a human specialist via the CRM. "
                "The specialist team has a 24-hour response SLA. Customers must "
                "raise refund requests within 30 days of delivery."
            ),
            "last_updated": "2025-04-01",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-002",
            "title": "Return Window Policy",
            "tags": ["return", "window", "policy", "days", "eligible"],
            "content": (
                "Customers may return items within 30 days of delivery. "
                "Electronics are eligible for return within 7 days. "
                "Items must be unused and in original packaging."
            ),
            "last_updated": "2025-03-15",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-003",
            "title": "Escalation SLA and Specialist Response Times",
            "tags": ["escalation", "sla", "specialist", "response", "24-hour"],
            "content": (
                "Escalated cases are assigned to a specialist within 2 hours. "
                "Specialists respond within 24 hours. High-priority cases "
                "(above Rs.25,000) are handled first."
            ),
            "last_updated": "2025-04-10",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-004",
            "title": "Partial Cancellation and Address Update Rules",
            "tags": ["cancel", "cancellation", "partial", "reship", "address", "update"],
            "content": (
                "Customers may cancel individual line items from orders in "
                "'placed' or 'processing' status. Shipped or delivered orders "
                "require a return request. Address updates are allowed for "
                "orders not yet shipped."
            ),
            "last_updated": "2025-04-05",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-005",
            "title": "Payment Methods and Refund Processing Times",
            "tags": ["payment", "refund", "methods", "sla", "days", "processing"],
            "content": (
                "Supported methods: HDFC Credit, ICICI Debit, SBI Net Banking, "
                "UPI, and COD. Refunds reflect within 5 business days of initiation."
            ),
            "last_updated": "2025-03-20",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
    ]


# ---------------------------------------------------------------------------
# Cases (pre-existing for history tests)
# ---------------------------------------------------------------------------

def build_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id":     "CASE-EXIST1",
            "customer_id": "CUST-001",
            "order_id":    "ORD-78323",
            "status":      "resolved",
            "priority":    "medium",
            "description": (
                "[ESCALATION CASE - AtlasCare]\n"
                "Customer  : CUST-001\nOrder     : ORD-78323\n"
                "Priority  : MEDIUM\nAmount    : N/A\n"
                "Reason    : Wrong size delivered.\n"
                "Action    : Requires specialist review.\n"
                "Trace ID  : trc-historical01"
            ),
            "amount_inr":  None,
            "trace_id":    "trc-historical01",
            "created_at":  _ts(15),
        },
    ]


# ---------------------------------------------------------------------------
# Payment config
# ---------------------------------------------------------------------------

def build_payment_config() -> dict[str, Any]:
    return {
        "auto_refund_limit_inr": 25000,
        "supported_methods": [
            "HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"
        ],
        "refund_sla_days": 5,
        "behaviour": {
            "failure_rate":    0.03,
            "failure_code":    "504",
            "failure_message": "PAYMENT_GATEWAY_TIMEOUT",
        },
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def build_sessions() -> list[dict[str, Any]]:
    return [
        {"session_id": "sess-cust001",  "customer_id": "CUST-001"},
        {"session_id": "sess-cust002",  "customer_id": "CUST-002"},
        {"session_id": "sess-cust003",  "customer_id": "CUST-003"},
        {"session_id": "sess-cust004",  "customer_id": "CUST-004"},
        {"session_id": "test-gold",     "customer_id": "CUST-001"},
        {"session_id": "test-platinum", "customer_id": "CUST-002"},
        {"session_id": "test-standard", "customer_id": "CUST-003"},
        {"session_id": "test-silver",   "customer_id": "CUST-004"},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "orders.json":         {"orders":    build_orders()},
        "crm_cases.json":      {"customers": build_customers(), "cases": build_cases()},
        "kb_articles.json":    {"articles":  build_kb_articles()},
        "payment_config.json": build_payment_config(),
        "sessions.json":       {"sessions":  build_sessions()},
        "refunds.json":        {"refunds":   []},
    }

    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"  [OK] {filename:<30} {_describe(data)}")

    print(f"\nData written to: {os.path.abspath(output_dir)}")


def _describe(data: dict) -> str:
    for key in ("orders", "articles", "sessions", "refunds"):
        if key in data:
            return f"{len(data[key])} {key}"
    if "customers" in data:
        return f"{len(data['customers'])} customers, {len(data.get('cases', []))} cases"
    return "config"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AtlasCare synthetic data.")
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory (default: ./data)",
    )
    args = parser.parse_args()
    print("AtlasCare Synthetic Data Generator")
    print("=" * 40)
    generate(args.output_dir)