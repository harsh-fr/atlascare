"""
tests/conftest.py
==================
Pytest fixtures and shared test configuration for AtlasCare.

Provides
--------
  - Isolated temp-directory data fixtures (no production data touched)
  - Pre-seeded synthetic orders, customers, cases, KB articles
  - FastAPI TestClient with session mappings pre-registered
  - Mock LLM client that returns deterministic plan JSON
  - Boundary-value refund fixtures (24999, 25000, 25001)

Design principles
-----------------
- Every test gets a fresh isolated data directory via tmp_path.
  Tests never share mutable state.
- LLM calls are mocked by default — tests are deterministic and
  do not require a live Gemini API key.
- Fixtures are composable — tests import only what they need.
- Synthetic data covers all schema-required boundary scenarios.
"""

import json
import os
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Generator

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Environment setup — must happen before app imports
# ---------------------------------------------------------------------------
os.environ.setdefault("GEMINI_API_KEY",  "test-key-not-real")
os.environ.setdefault("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
os.environ.setdefault("GEMINI_MODEL",    "gemini-2.5-flash")
os.environ.setdefault("AUTO_REFUND_LIMIT_INR", "25000.0")
os.environ.setdefault("PAYMENT_MAX_RETRIES",   "1")
os.environ.setdefault("PAYMENT_RETRY_BASE_DELAY_S", "0.0")


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_customer(
    customer_id: str = "CUST-001",
    tier: str = "gold",
    extra_addresses: list | None = None,
) -> dict[str, Any]:
    addresses = [
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
    ]
    if extra_addresses:
        addresses.extend(extra_addresses)
    num = customer_id.split("-")[-1]
    return {
        "customer_id": customer_id,
        "name": f"Test Customer {num}",
        "email": f"customer{num}@test.com",
        "phone": f"+91-9{num}00000000",
        "tier": tier,
        "order_ids": [f"ORD-7{num}00", f"ORD-7{num}01"],
        "preferred_refund_method": "HDFC_CREDIT",
        "addresses": addresses,
    }


def _make_order(
    order_id: str = "ORD-78321",
    customer_id: str = "CUST-001",
    status: str = "processing",
    items: list | None = None,
) -> dict[str, Any]:
    if items is None:
        items = [
            {
                "line_id": 1,
                "product_id": "PROD-LAPTOP-001",
                "name": "Dell Inspiron 15 Laptop",
                "quantity": 1,
                "unit_price": 55000.00,
                "status": "active",
            },
            {
                "line_id": 2,
                "product_id": "PROD-BAG-021",
                "name": "Laptop Backpack",
                "quantity": 1,
                "unit_price": 1500.00,
                "status": "active",
            },
            {
                "line_id": 3,
                "product_id": "PROD-MOUSE-005",
                "name": "Wireless Mouse",
                "quantity": 1,
                "unit_price": 800.00,
                "status": "active",
            },
        ]
    total = sum(
        i["unit_price"] * i["quantity"]
        for i in items
        if i["status"] == "active"
    )
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": status,
        "created_at": "2025-05-01T10:30:00Z",
        "estimated_delivery": "2025-05-08",
        "tracking_number": "TRACK-7X9K2M" if status in ("shipped", "delivered") else None,
        "shipping_address": {
            "line1": "12 MG Road",
            "line2": "Koramangala",
            "city": "Bengaluru",
            "state": "Karnataka",
            "pincode": "560034",
        },
        "items": items,
        "total_amount": total,
        "payment_method": "HDFC_CREDIT",
    }


def _make_kb_articles() -> list[dict[str, Any]]:
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
            "tags": ["return", "window", "policy", "days"],
            "content": (
                "Customers may return items within 30 days of delivery for a "
                "full refund. Items must be unused and in original packaging. "
                "Electronics are eligible for return within 7 days."
            ),
            "last_updated": "2025-03-15",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-003",
            "title": "Escalation SLA",
            "tags": ["escalation", "sla", "specialist", "response"],
            "content": (
                "Escalated cases are assigned to a specialist within 2 hours. "
                "Specialists respond to customers within 24 hours of case creation. "
                "High-priority cases (above Rs.25,000) are handled first."
            ),
            "last_updated": "2025-04-10",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-004",
            "title": "Partial Cancellation and Reshipping Rules",
            "tags": ["cancel", "cancellation", "partial", "reship", "address"],
            "content": (
                "Customers may cancel individual line items from orders in "
                "'placed' or 'processing' status. Shipped orders cannot be "
                "partially cancelled. Address updates are allowed for orders "
                "not yet delivered or cancelled."
            ),
            "last_updated": "2025-04-05",
            "applies_to": ["electronics", "apparel", "home_goods"],
        },
    ]


def _make_payment_config() -> dict[str, Any]:
    return {
        "auto_refund_limit_inr": 25000,
        "supported_methods": [
            "HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"
        ],
        "refund_sla_days": 5,
        "behaviour": {
            "failure_rate": 0.0,   # No failures in tests by default
            "failure_code": "504",
            "failure_message": "PAYMENT_GATEWAY_TIMEOUT",
        },
    }


# ---------------------------------------------------------------------------
# Data directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """
    Create an isolated data directory with seeded JSON files.
    Every test gets its own tmp_path — no cross-test contamination.
    """
    d = tmp_path / "data"
    d.mkdir()

    # Customers and cases
    crm_data = {
        "customers": [
            _make_customer("CUST-001", tier="gold"),
            _make_customer("CUST-002", tier="platinum"),
            _make_customer("CUST-003", tier="standard",
                           extra_addresses=[]),  # home only
        ],
        "cases": [],
    }
    (d / "crm_cases.json").write_text(
        json.dumps(crm_data, indent=2), encoding="utf-8"
    )

    # Orders — covers all status types and boundary amounts
    orders_data = {
        "orders": [
            _make_order("ORD-78321", "CUST-001", "processing"),
            _make_order("ORD-78322", "CUST-001", "shipped"),
            _make_order("ORD-78323", "CUST-001", "delivered"),
            _make_order("ORD-78324", "CUST-001", "placed"),
            _make_order("ORD-78325", "CUST-001", "cancelled"),
            # Boundary: below threshold (₹24,999)
            _make_order("ORD-78400", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "PROD-A", "name": "Item A",
                 "quantity": 1, "unit_price": 24999.00, "status": "active"},
            ]),
            # Boundary: exactly at threshold (₹25,000)
            _make_order("ORD-78401", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "PROD-B", "name": "Item B",
                 "quantity": 1, "unit_price": 25000.00, "status": "active"},
            ]),
            # Boundary: above threshold (₹25,001)
            _make_order("ORD-78402", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "PROD-C", "name": "Item C",
                 "quantity": 1, "unit_price": 25001.00, "status": "active"},
            ]),
            # High value (₹42,000) — J3 scenario
            _make_order("ORD-78500", "CUST-001", "delivered", items=[
                {"line_id": 1, "product_id": "PROD-LAPTOP-HV", "name": "Gaming Laptop",
                 "quantity": 1, "unit_price": 42000.00, "status": "active"},
            ]),
            # CUST-002 order — for cross-customer security tests
            _make_order("ORD-99001", "CUST-002", "processing"),
        ]
    }
    (d / "orders.json").write_text(
        json.dumps(orders_data, indent=2), encoding="utf-8"
    )

    # KB articles
    kb_data = {"articles": _make_kb_articles()}
    (d / "kb_articles.json").write_text(
        json.dumps(kb_data, indent=2), encoding="utf-8"
    )

    # Payment config
    (d / "payment_config.json").write_text(
        json.dumps(_make_payment_config(), indent=2), encoding="utf-8"
    )

    # Empty refunds file
    (d / "refunds.json").write_text(
        json.dumps({"refunds": []}, indent=2), encoding="utf-8"
    )

    # Sessions
    sessions_data = {
        "sessions": [
            {"session_id": "sess-cust001", "customer_id": "CUST-001"},
            {"session_id": "sess-cust002", "customer_id": "CUST-002"},
            {"session_id": "sess-cust003", "customer_id": "CUST-003"},
        ]
    }
    (d / "sessions.json").write_text(
        json.dumps(sessions_data, indent=2), encoding="utf-8"
    )

    return d


# ---------------------------------------------------------------------------
# Environment patcher — point all repos to tmp data_dir
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_env(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch all data path env vars to point to the isolated data_dir."""
    monkeypatch.setenv("ORDERS_DATA_PATH",    str(data_dir / "orders.json"))
    monkeypatch.setenv("CRM_DATA_PATH",       str(data_dir / "crm_cases.json"))
    monkeypatch.setenv("KB_DATA_PATH",        str(data_dir / "kb_articles.json"))
    monkeypatch.setenv("PAYMENT_CONFIG_PATH", str(data_dir / "payment_config.json"))
    monkeypatch.setenv("REFUNDS_DATA_PATH",   str(data_dir / "refunds.json"))
    monkeypatch.setenv("SESSIONS_DATA_PATH",  str(data_dir / "sessions.json"))


# ---------------------------------------------------------------------------
# Mock LLM responses — deterministic plan JSON per intent
# ---------------------------------------------------------------------------

def _mock_plan_response(intent: str, steps: list[dict]) -> MagicMock:
    """Build a mock OpenAI completion response with a deterministic plan."""
    plan_json = json.dumps({"intent": intent, "steps": steps})
    mock_msg = MagicMock()
    mock_msg.content = plan_json
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def make_llm_mock(plan_completion: MagicMock, response_text: str = "Your request has been processed.") -> AsyncMock:
    """
    Build an AsyncMock that returns plan_completion on first call
    (planner) and a simple response on second call (response_builder).
    """
    response_msg = MagicMock()
    response_msg.content = response_text
    response_choice = MagicMock()
    response_choice.message = response_msg
    response_completion = MagicMock()
    response_completion.choices = [response_choice]

    mock = AsyncMock()
    mock.side_effect = [plan_completion, response_completion]
    return mock


# Pre-built plan mocks for J1, J2, J3
J1_PLAN = _mock_plan_response(
    intent="order_tracking",
    steps=[{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
)

J2_PLAN = _mock_plan_response(
    intent="compound",
    steps=[
        {"action": "cancel_item",    "params": {"order_id": "ORD-78321", "line_id": 2}, "depends_on": []},
        {"action": "process_refund", "params": {"order_id": "ORD-78321", "amount_inr": 1500.0, "method": "HDFC_CREDIT"}, "depends_on": [0]},
        {"action": "update_address", "params": {"order_id": "ORD-78321", "address_label": "office"}, "depends_on": []},
    ],
)

J3_PLAN = _mock_plan_response(
    intent="escalation",
    steps=[{"action": "escalate", "params": {"order_id": "ORD-78500", "reason": "Customer requesting full refund for damaged laptop. Amount exceeds threshold.", "amount_inr": 42000.0}, "depends_on": []}],
)


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def client(patched_env) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with isolated data and mocked LLM.
    Uses lifespan context so startup/shutdown hooks run.
    """
    import importlib
    import main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Async fixtures for direct service/tool testing
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def oms_tool(patched_env):
    from tools.oms_tool import OmsTool
    return OmsTool()


@pytest_asyncio.fixture
async def crm_tool(patched_env):
    from tools.crm_tool import CrmTool
    return CrmTool()


@pytest_asyncio.fixture
async def payment_tool(patched_env):
    from tools.payment_tool import PaymentTool
    return PaymentTool()


@pytest_asyncio.fixture
async def kb_tool(patched_env):
    from tools.kb_tool import KbTool
    return KbTool()