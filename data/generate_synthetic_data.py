"""
data/generate_synthetic_data.py
================================
Synthetic data generator for AtlasCare — Acme Retail Co.

Generates all JSON data files conforming strictly to the provided schemas:
  - data/orders.json
  - data/crm_cases.json
  - data/kb_articles.json
  - data/payment_config.json
  - data/sessions.json
  - data/refunds.json  (always empty — populated at runtime)

DOES NOT touch data/users.json (auth file, hashed passwords).

Coverage strategy
-----------------
Every record is purposefully designed to exercise a specific scenario:

  Refund boundary tests (auto_refund_limit = Rs.25,000)
    ORD-10005  Rs.24,999  just below threshold
    ORD-10006  Rs.25,000  exactly at threshold
    ORD-10007  Rs.25,001  just above threshold — must escalate
    ORD-10008  Rs.42,000  high-value damaged item — J3 + GR-004

  Order-status coverage
    placed, processing, shipped, delivered, cancelled

  COD scenarios
    ORD-10010  processing  COD multi-item (cancel = no refund)
    ORD-10011  delivered   COD (return needs electronic method)
    ORD-10012  shipped     COD (cannot cancel shipped)
    ORD-10013  placed      COD multi-item (partial cancel test)
    ORD-20003  delivered   COD Rs.82,000 (high-value escalation)

  Month-based filter tests (CUST-001, 14 orders across Feb/Mar/Apr/May 2026)
    May 2026   ORD-10001 to ORD-10005 + ORD-10010 to ORD-10013  (~10 orders)
    April 2026 ORD-10006 to ORD-10008                             (3 orders)
    March 2026 ORD-10009                                          (1 order)
    Feb 2026   ORD-10014                                          (1 order)

  Guardrail GR-004
    ORD-10008 has an open CRM case (CASE-OPEN01). Attempting to call
    process_refund AND escalate on it in the same turn triggers post-guardrail.

  Customer/address diversity
    CUST-001  home + office (Bengaluru)
    CUST-002  home + office (Mumbai)
    CUST-003  home ONLY (Chennai) — tests missing label scenario
    CUST-004  home + office + parents (Noida / Gurugram / Allahabad)

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
    """Return ISO-8601 timestamp (UTC) for `days_ago` days before now."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _date(days_from_now: int = 0) -> str:
    """Return YYYY-MM-DD date for `days_from_now` days from now."""
    dt = datetime.now(timezone.utc) + timedelta(days=days_from_now)
    return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Shared shipping address objects
# ---------------------------------------------------------------------------

_HOME_BLR = {
    "line1": "12 MG Road, Koramangala",
    "city": "Bengaluru",
    "state": "Karnataka",
    "pincode": "560034",
}
_HOME_MUM = {
    "line1": "Flat 3B, Sunshine Apartments, Bandra West",
    "city": "Mumbai",
    "state": "Maharashtra",
    "pincode": "400050",
}
_HOME_CHN = {
    "line1": "22 Anna Salai, Teynampet",
    "city": "Chennai",
    "state": "Tamil Nadu",
    "pincode": "600018",
}
_HOME_NOI = {
    "line1": "B-45, Sector 62",
    "city": "Noida",
    "state": "Uttar Pradesh",
    "pincode": "201301",
}


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
                "ORD-10001", "ORD-10002", "ORD-10003", "ORD-10004", "ORD-10005",
                "ORD-10006", "ORD-10007", "ORD-10008", "ORD-10009", "ORD-10010",
                "ORD-10011", "ORD-10012", "ORD-10013", "ORD-10014",
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
            "order_ids": ["ORD-20001", "ORD-20002", "ORD-20003"],
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
                    "line1": "WeWork BKC, Bandra Kurla Complex",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "pincode": "400051",
                },
            ],
        },
        {
            # No office address — tests missing address label scenario
            "customer_id": "CUST-003",
            "name": "Divya Nair",
            "email": "divya.nair@email.com",
            "phone": "+91-9988776655",
            "tier": "standard",
            "order_ids": ["ORD-30001", "ORD-30002"],
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
            "order_ids": ["ORD-40001", "ORD-40002"],
            "preferred_refund_method": "ICICI_DEBIT",
            "addresses": [
                {
                    "label": "home",
                    "line1": "B-45, Sector 62",
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

    # ── CUST-001 — MAY 2026 GROUP (recent ~0–22 days ago) ──────────────────

    # ORD-10001: placed, 3 items all active, UPI, Rs.3,250
    ord_10001 = {
        "order_id": "ORD-10001", "customer_id": "CUST-001",
        "status": "placed",
        "created_at": _ts(1), "estimated_delivery": _date(6),
        "tracking_number": None,
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-BOOK-101",
             "name": "Python Crash Course Book",
             "quantity": 1, "unit_price": 750.00, "status": "active"},
            {"line_id": 2, "product_id": "PROD-HUB-022",
             "name": "USB-C 7-in-1 Hub",
             "quantity": 1, "unit_price": 1500.00, "status": "active"},
            {"line_id": 3, "product_id": "PROD-DESK-009",
             "name": "Bamboo Desk Organizer",
             "quantity": 1, "unit_price": 1000.00, "status": "active"},
        ],
        "total_amount": 3250.00,
        "payment_method": "UPI",
    }

    # ORD-10002: processing, 3 items (item 2 pre-cancelled), HDFC_CREDIT, Rs.18,200
    ord_10002 = {
        "order_id": "ORD-10002", "customer_id": "CUST-001",
        "status": "processing",
        "created_at": _ts(4), "estimated_delivery": _date(3),
        "tracking_number": None,
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-WM-007",
             "name": "Bosch Front Load Washing Machine",
             "quantity": 1, "unit_price": 18000.00, "status": "active"},
            {"line_id": 2, "product_id": "PROD-WM-STAND-001",
             "name": "Machine Installation Stand",
             "quantity": 1, "unit_price": 500.00, "status": "cancelled"},
            {"line_id": 3, "product_id": "PROD-LAUNDRY-003",
             "name": "Laundry Mesh Bag",
             "quantity": 1, "unit_price": 200.00, "status": "active"},
        ],
        "total_amount": 18200.00,
        "payment_method": "HDFC_CREDIT",
    }

    # ORD-10003: shipped, 1 item, ICICI_DEBIT, Rs.8,999
    ord_10003 = {
        "order_id": "ORD-10003", "customer_id": "CUST-001",
        "status": "shipped",
        "created_at": _ts(8), "estimated_delivery": _date(1),
        "tracking_number": "TRACK-A1B2C3",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-SHOE-042",
             "name": "Nike Air Force 1 Sneakers",
             "quantity": 1, "unit_price": 8999.00, "status": "active"},
        ],
        "total_amount": 8999.00,
        "payment_method": "ICICI_DEBIT",
    }

    # ORD-10004: delivered, 1 item, HDFC_CREDIT, Rs.12,000
    ord_10004 = {
        "order_id": "ORD-10004", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(15), "estimated_delivery": _date(-8),
        "tracking_number": "TRACK-D4E5F6",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-AP-011",
             "name": "Dyson Hot+Cool Air Purifier",
             "quantity": 1, "unit_price": 12000.00, "status": "active"},
        ],
        "total_amount": 12000.00,
        "payment_method": "HDFC_CREDIT",
    }

    # ORD-10005: delivered, 1 item, UPI, Rs.24,999 — just below refund threshold
    ord_10005 = {
        "order_id": "ORD-10005", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(22), "estimated_delivery": _date(-14),
        "tracking_number": "TRACK-G7H8I9",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-WATCH-009",
             "name": "Apple Watch Series 9 (45mm)",
             "quantity": 1, "unit_price": 24999.00, "status": "active"},
        ],
        "total_amount": 24999.00,
        "payment_method": "UPI",
    }

    # ORD-10010: processing, COD, 3 items all active, Rs.6,800 — cancel = no refund
    ord_10010 = {
        "order_id": "ORD-10010", "customer_id": "CUST-001",
        "status": "processing",
        "created_at": _ts(2), "estimated_delivery": _date(4),
        "tracking_number": None,
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-FAN-018",
             "name": "Havells 1200mm Ceiling Fan",
             "quantity": 1, "unit_price": 2500.00, "status": "active"},
            {"line_id": 2, "product_id": "PROD-IRON-005",
             "name": "Philips EasySpeed Steam Iron",
             "quantity": 1, "unit_price": 1299.00, "status": "active"},
            {"line_id": 3, "product_id": "PROD-STAB-002",
             "name": "V-Guard Voltage Stabilizer",
             "quantity": 1, "unit_price": 3001.00, "status": "active"},
        ],
        "total_amount": 6800.00,
        "payment_method": "COD",
    }

    # ORD-10011: delivered, COD, 1 item, Rs.8,999 — return needs electronic method
    ord_10011 = {
        "order_id": "ORD-10011", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(19), "estimated_delivery": _date(-11),
        "tracking_number": "TRACK-COD01",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-SHOE-043",
             "name": "Nike Air Max 270",
             "quantity": 1, "unit_price": 8999.00, "status": "active"},
        ],
        "total_amount": 8999.00,
        "payment_method": "COD",
    }

    # ORD-10012: shipped, COD, 1 item, Rs.2,499 — cannot cancel shipped
    ord_10012 = {
        "order_id": "ORD-10012", "customer_id": "CUST-001",
        "status": "shipped",
        "created_at": _ts(5), "estimated_delivery": _date(2),
        "tracking_number": "TRACK-COD02",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-WATCH-002",
             "name": "Fastrack Analog Watch",
             "quantity": 1, "unit_price": 2499.00, "status": "active"},
        ],
        "total_amount": 2499.00,
        "payment_method": "COD",
    }

    # ORD-10013: placed, COD, 4 items, Rs.4,396 — partial cancel test (cancel Juttis only)
    ord_10013 = {
        "order_id": "ORD-10013", "customer_id": "CUST-001",
        "status": "placed",
        "created_at": _ts(0), "estimated_delivery": _date(7),
        "tracking_number": None,
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-KURTA-BL",
             "name": "Cotton Kurta Blue",
             "quantity": 1, "unit_price": 899.00, "status": "active"},
            {"line_id": 2, "product_id": "PROD-KURTA-GR",
             "name": "Cotton Kurta Green",
             "quantity": 1, "unit_price": 899.00, "status": "active"},
            {"line_id": 3, "product_id": "PROD-DUP-001",
             "name": "Banarasi Dupatta",
             "quantity": 1, "unit_price": 1299.00, "status": "active"},
            {"line_id": 4, "product_id": "PROD-JUTTI-001",
             "name": "Kolhapuri Juttis",
             "quantity": 1, "unit_price": 1299.00, "status": "active"},
        ],
        "total_amount": 4396.00,
        "payment_method": "COD",
    }

    # ── CUST-001 — APRIL 2026 GROUP (35–50 days ago) ───────────────────────

    # ORD-10006: delivered, HDFC_CREDIT, Rs.25,000 — exactly at threshold boundary
    ord_10006 = {
        "order_id": "ORD-10006", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(35), "estimated_delivery": _date(-27),
        "tracking_number": "TRACK-J0K1L2",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-PHONE-031",
             "name": "Samsung Galaxy S24 (256GB)",
             "quantity": 1, "unit_price": 25000.00, "status": "active"},
        ],
        "total_amount": 25000.00,
        "payment_method": "HDFC_CREDIT",
    }

    # ORD-10007: delivered, ICICI_DEBIT, Rs.25,001 — just above threshold, must escalate
    ord_10007 = {
        "order_id": "ORD-10007", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(42), "estimated_delivery": _date(-34),
        "tracking_number": "TRACK-M3N4O5",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-TAB-011",
             "name": "iPad Pro 11-inch (M4)",
             "quantity": 1, "unit_price": 25001.00, "status": "active"},
        ],
        "total_amount": 25001.00,
        "payment_method": "ICICI_DEBIT",
    }

    # ORD-10008: delivered, HDFC_CREDIT, Rs.42,000 — high-value, open CRM case (GR-004 test)
    ord_10008 = {
        "order_id": "ORD-10008", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(50), "estimated_delivery": _date(-42),
        "tracking_number": "TRACK-P6Q7R8",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-LAPTOP-XPS",
             "name": "Dell XPS 13 Plus Laptop",
             "quantity": 1, "unit_price": 42000.00, "status": "active"},
        ],
        "total_amount": 42000.00,
        "payment_method": "HDFC_CREDIT",
    }

    # ── CUST-001 — MARCH 2026 GROUP (62 days ago) ──────────────────────────

    # ORD-10009: cancelled, SBI_NETBANKING, total Rs.0 (item cancelled)
    ord_10009 = {
        "order_id": "ORD-10009", "customer_id": "CUST-001",
        "status": "cancelled",
        "created_at": _ts(62), "estimated_delivery": _date(-52),
        "tracking_number": None,
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-CHAIR-007",
             "name": "Ergonomic Mesh Office Chair",
             "quantity": 1, "unit_price": 18000.00, "status": "cancelled"},
        ],
        "total_amount": 0.00,
        "payment_method": "SBI_NETBANKING",
    }

    # ── CUST-001 — FEBRUARY 2026 GROUP (92 days ago) ───────────────────────

    # ORD-10014: delivered, HDFC_CREDIT, Rs.55,000 — fraud test ("I never placed this")
    ord_10014 = {
        "order_id": "ORD-10014", "customer_id": "CUST-001",
        "status": "delivered",
        "created_at": _ts(92), "estimated_delivery": _date(-83),
        "tracking_number": "TRACK-FEB01",
        "shipping_address": _HOME_BLR,
        "items": [
            {"line_id": 1, "product_id": "PROD-LAPTOP-M3",
             "name": "Apple MacBook Air M3 (16GB)",
             "quantity": 1, "unit_price": 55000.00, "status": "active"},
        ],
        "total_amount": 55000.00,
        "payment_method": "HDFC_CREDIT",
    }

    # ── CUST-002 ────────────────────────────────────────────────────────────

    # ORD-20001: processing, UPI, Rs.29,000 — cross-customer ownership test
    ord_20001 = {
        "order_id": "ORD-20001", "customer_id": "CUST-002",
        "status": "processing",
        "created_at": _ts(3), "estimated_delivery": _date(4),
        "tracking_number": None,
        "shipping_address": _HOME_MUM,
        "items": [
            {"line_id": 1, "product_id": "PROD-HP-031",
             "name": "Sony WH-1000XM5 Headphones",
             "quantity": 1, "unit_price": 29000.00, "status": "active"},
        ],
        "total_amount": 29000.00,
        "payment_method": "UPI",
    }

    # ORD-20002: shipped, ICICI_DEBIT, Rs.19,000 (2x Chanel perfume Rs.9,500 each)
    ord_20002 = {
        "order_id": "ORD-20002", "customer_id": "CUST-002",
        "status": "shipped",
        "created_at": _ts(6), "estimated_delivery": _date(1),
        "tracking_number": "TRACK-MUM01",
        "shipping_address": _HOME_MUM,
        "items": [
            {"line_id": 1, "product_id": "PROD-PERF-011",
             "name": "Chanel No.5 Perfume 100ml",
             "quantity": 2, "unit_price": 9500.00, "status": "active"},
        ],
        "total_amount": 19000.00,
        "payment_method": "ICICI_DEBIT",
    }

    # ORD-20003: delivered, COD, Rs.82,000 — high-value COD escalation
    ord_20003 = {
        "order_id": "ORD-20003", "customer_id": "CUST-002",
        "status": "delivered",
        "created_at": _ts(25), "estimated_delivery": _date(-17),
        "tracking_number": "TRACK-MUM02",
        "shipping_address": _HOME_MUM,
        "items": [
            {"line_id": 1, "product_id": "PROD-TV-065",
             "name": "Sony Bravia 65-inch OLED TV",
             "quantity": 1, "unit_price": 82000.00, "status": "active"},
        ],
        "total_amount": 82000.00,
        "payment_method": "COD",
    }

    # ── CUST-003 ────────────────────────────────────────────────────────────

    # ORD-30001: delivered, SBI_NETBANKING, Rs.4,500
    ord_30001 = {
        "order_id": "ORD-30001", "customer_id": "CUST-003",
        "status": "delivered",
        "created_at": _ts(30), "estimated_delivery": _date(-22),
        "tracking_number": "TRACK-CHN01",
        "shipping_address": _HOME_CHN,
        "items": [
            {"line_id": 1, "product_id": "PROD-SAREE-055",
             "name": "Kanjeevaram Silk Saree",
             "quantity": 1, "unit_price": 4500.00, "status": "active"},
        ],
        "total_amount": 4500.00,
        "payment_method": "SBI_NETBANKING",
    }

    # ORD-30002: placed, COD, 4 items, Rs.3,396 — no office address test for CUST-003
    ord_30002 = {
        "order_id": "ORD-30002", "customer_id": "CUST-003",
        "status": "placed",
        "created_at": _ts(0), "estimated_delivery": _date(7),
        "tracking_number": None,
        "shipping_address": _HOME_CHN,
        "items": [
            {"line_id": 1, "product_id": "PROD-KURTI-XL",
             "name": "Ethnic Kurti (XL)",
             "quantity": 1, "unit_price": 1499.00, "status": "active"},
            {"line_id": 2, "product_id": "PROD-PALAZZO-001",
             "name": "Palazzo Pants",
             "quantity": 1, "unit_price": 999.00, "status": "active"},
            {"line_id": 3, "product_id": "PROD-STOLE-001",
             "name": "Silk Stole",
             "quantity": 1, "unit_price": 499.00, "status": "active"},
            {"line_id": 4, "product_id": "PROD-ANKLET-001",
             "name": "Oxidised Anklets",
             "quantity": 1, "unit_price": 399.00, "status": "active"},
        ],
        "total_amount": 3396.00,
        "payment_method": "COD",
    }

    # ── CUST-004 ────────────────────────────────────────────────────────────

    # ORD-40001: processing, ICICI_DEBIT, Rs.35,000 — above threshold
    ord_40001 = {
        "order_id": "ORD-40001", "customer_id": "CUST-004",
        "status": "processing",
        "created_at": _ts(2), "estimated_delivery": _date(5),
        "tracking_number": None,
        "shipping_address": _HOME_NOI,
        "items": [
            {"line_id": 1, "product_id": "PROD-FRIDGE-002",
             "name": "LG 260L Double Door Refrigerator",
             "quantity": 1, "unit_price": 35000.00, "status": "active"},
        ],
        "total_amount": 35000.00,
        "payment_method": "ICICI_DEBIT",
    }

    # ORD-40002: placed, ICICI_DEBIT, Rs.38,500 — multi-item, above threshold
    ord_40002 = {
        "order_id": "ORD-40002", "customer_id": "CUST-004",
        "status": "placed",
        "created_at": _ts(0), "estimated_delivery": _date(7),
        "tracking_number": None,
        "shipping_address": _HOME_NOI,
        "items": [
            {"line_id": 1, "product_id": "PROD-AC-018",
             "name": "Daikin 1.5 Ton 5-Star AC",
             "quantity": 1, "unit_price": 38000.00, "status": "active"},
            {"line_id": 2, "product_id": "PROD-COVER-009",
             "name": "AC Dust Cover",
             "quantity": 1, "unit_price": 500.00, "status": "active"},
        ],
        "total_amount": 38500.00,
        "payment_method": "ICICI_DEBIT",
    }

    return [
        # CUST-001 — May 2026
        ord_10001, ord_10002, ord_10003, ord_10004, ord_10005,
        ord_10010, ord_10011, ord_10012, ord_10013,
        # CUST-001 — April 2026
        ord_10006, ord_10007, ord_10008,
        # CUST-001 — March 2026
        ord_10009,
        # CUST-001 — February 2026
        ord_10014,
        # CUST-002
        ord_20001, ord_20002, ord_20003,
        # CUST-003
        ord_30001, ord_30002,
        # CUST-004
        ord_40001, ord_40002,
    ]


# ---------------------------------------------------------------------------
# KB articles  (7 articles)
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
                "raise refund requests within 30 days of delivery. "
                "The boundary is strictly greater than Rs.25,000 — orders of exactly "
                "Rs.25,000 are eligible for auto-refund."
            ),
            "last_updated": "2026-04-01",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-002",
            "title": "Return Window Policy",
            "tags": ["return", "window", "policy", "days", "eligible"],
            "content": (
                "Customers may return items within 30 days of delivery. "
                "Electronics are eligible for return within 7 days. "
                "Items must be unused and in original packaging. "
                "For orders in 'placed' or 'processing' status, use the cancellation flow instead. "
                "For shipped or delivered orders, a return/pickup will be arranged."
            ),
            "last_updated": "2026-03-15",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-003",
            "title": "Escalation SLA and Specialist Response Times",
            "tags": ["escalation", "sla", "specialist", "response", "24-hour"],
            "content": (
                "Escalated cases are assigned to a specialist within 2 hours. "
                "Specialists respond within 24 hours. High-priority cases "
                "(above Rs.25,000, damaged goods, fraud, or legal action) are handled first. "
                "Customers will receive an email confirmation once a case is opened."
            ),
            "last_updated": "2026-04-10",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-004",
            "title": "Partial Cancellation and Address Update Rules",
            "tags": ["cancel", "cancellation", "partial", "reship", "address", "update"],
            "content": (
                "Customers may cancel individual line items from orders in "
                "'placed' or 'processing' status. Shipped or delivered orders "
                "require a return request — cancellation is not possible once dispatched. "
                "Address updates are allowed only for orders not yet shipped (placed or processing). "
                "Once an order is shipped, the address cannot be changed."
            ),
            "last_updated": "2026-04-05",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-005",
            "title": "Payment Methods and Refund Processing Times",
            "tags": ["payment", "refund", "methods", "sla", "days", "processing"],
            "content": (
                "Supported refund methods: HDFC Credit Card, ICICI Debit Card, "
                "SBI Net Banking, UPI (GPay, PhonePe, Paytm all map to UPI), and "
                "original payment method. Refunds reflect within 5 business days of initiation. "
                "COD orders cannot be refunded via cash — an electronic method must be provided."
            ),
            "last_updated": "2026-03-20",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-006",
            "title": "Cash on Delivery (COD) Refund Policy",
            "tags": ["cod", "cash", "delivery", "refund", "electronic"],
            "content": (
                "For Cash on Delivery orders that are cancelled before dispatch, no refund is "
                "needed as payment was never collected. "
                "For delivered COD orders being returned, the refund CANNOT be issued as cash. "
                "Customers must provide an electronic refund method: UPI (GPay, PhonePe, Paytm), "
                "HDFC Credit Card, ICICI Debit Card, or SBI Net Banking. "
                "Asking to refund 'the same way' or 'in cash' for a COD order is not accepted — "
                "the agent must prompt for a valid electronic method. "
                "High-value COD returns above Rs.25,000 follow the standard escalation path."
            ),
            "last_updated": "2026-04-15",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-007",
            "title": "Fraud and Unauthorized Orders",
            "tags": ["fraud", "unauthorized", "security", "account", "escalation"],
            "content": (
                "If a customer reports an order they did not place, or suspects their account "
                "has been accessed without authorization, this is treated as a high-priority "
                "security escalation. The agent must: (1) create a CRM escalation case with "
                "priority HIGH, (2) NOT take any other action on the order (no cancellation, "
                "no refund), (3) advise the customer to change their password immediately, "
                "and (4) inform them that the security team will respond within 24 hours. "
                "Phrases like 'I never placed this', 'I did not order this', 'someone else "
                "ordered this', or 'my account was hacked' all trigger this flow."
            ),
            "last_updated": "2026-04-20",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
    ]


# ---------------------------------------------------------------------------
# CRM Cases (pre-existing for history tests)
# ---------------------------------------------------------------------------

def build_cases() -> list[dict[str, Any]]:
    return [
        {
            # CASE-HIST01: resolved, medium priority — wrong size delivered
            "case_id":     "CASE-HIST01",
            "customer_id": "CUST-001",
            "order_id":    "ORD-10003",
            "status":      "resolved",
            "priority":    "medium",
            "description": (
                "[ESCALATION CASE - AtlasCare]\n"
                "Customer  : CUST-001\n"
                "Order     : ORD-10003\n"
                "Priority  : MEDIUM\n"
                "Amount    : N/A\n"
                "Reason    : Reported wrong size delivered.\n"
                "Action    : Requires specialist review.\n"
                "Trace ID  : trc-historical01"
            ),
            "amount_inr":  None,
            "trace_id":    "trc-historical01",
            "created_at":  _ts(20),
        },
        {
            # CASE-OPEN01: open, high priority — damaged Dell XPS laptop on delivery
            # Used in GR-004 test: trying to call process_refund AND escalate on ORD-10008
            # in the same turn triggers the post-guardrail critical block.
            "case_id":     "CASE-OPEN01",
            "customer_id": "CUST-001",
            "order_id":    "ORD-10008",
            "status":      "open",
            "priority":    "high",
            "description": (
                "[ESCALATION CASE - AtlasCare]\n"
                "Customer  : CUST-001\n"
                "Order     : ORD-10008\n"
                "Priority  : HIGH\n"
                "Amount    : 42000.00\n"
                "Reason    : Customer reported damaged laptop on delivery.\n"
                "Action    : Specialist investigation in progress.\n"
                "Trace ID  : trc-historical02"
            ),
            "amount_inr":  42000.00,
            "trace_id":    "trc-historical02",
            "created_at":  _ts(10),
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
# Main generate function
# ---------------------------------------------------------------------------

def _describe(data: dict) -> str:
    for key in ("orders", "articles", "sessions", "refunds"):
        if key in data:
            return f"{len(data[key])} {key}"
    if "customers" in data:
        return (
            f"{len(data['customers'])} customers, "
            f"{len(data.get('cases', []))} cases"
        )
    return "config"


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
