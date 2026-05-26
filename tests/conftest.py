"""
tests/conftest.py
==================
Pytest fixtures and shared test configuration for AtlasCare.

Root cause fix for failing tests
---------------------------------
The original conftest.make_llm_mock() returned a mock whose
side_effect was a list of completion OBJECTS.  But Planner._call_llm()
returns a raw string (the completion text), not a completion object.
So every test that used make_llm_mock got an empty tool_calls list
because the planner received a MagicMock object instead of a JSON string
and raised PlannerError internally, which the orchestrator swallowed.

Fix: make_llm_mock() now returns an AsyncMock whose side_effect is a
list of STRINGS:
  [plan_json_string, response_text_string]

The first call (planner) returns the plan JSON string.
The second call (response_builder) returns the response text string.
Both _call_llm methods return raw strings so this matches exactly.
"""

import hashlib
import json
import os
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Generator

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Environment — must be set before any app imports
# ---------------------------------------------------------------------------
os.environ.setdefault("GEMINI_API_KEY",             "test-key-not-real")
os.environ.setdefault("GEMINI_BASE_URL",            "https://generativelanguage.googleapis.com/v1beta/openai")
os.environ.setdefault("GEMINI_MODEL",               "gemini-2.5-flash")
os.environ.setdefault("AUTO_REFUND_LIMIT_INR",      "25000.0")
os.environ.setdefault("PAYMENT_MAX_RETRIES",        "1")
os.environ.setdefault("PAYMENT_RETRY_BASE_DELAY_S", "0.0")


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

def _make_user(
    username: str,
    customer_id: str,
    email: str | None = None,
    password: str = "Atlas@123",
) -> dict[str, Any]:
    return {
        "user_id":       f"USER-{username.upper()}",
        "username":      username,
        "email":         email or f"{username}@test.com",
        "password_hash": hashlib.sha256(password.encode()).hexdigest(),
        "customer_id":   customer_id,
        "created_at":    "2025-01-01T00:00:00Z",
    }


def _make_customer(
    customer_id: str = "CUST-001",
    tier: str = "gold",
    extra_addresses: list | None = None,
) -> dict[str, Any]:
    addresses = [
        {
            "label":   "home",
            "line1":   "12 MG Road, Koramangala",
            "city":    "Bengaluru",
            "state":   "Karnataka",
            "pincode": "560034",
        },
        {
            "label":   "office",
            "line1":   "4th Floor, Prestige Tower, Outer Ring Road",
            "city":    "Bengaluru",
            "state":   "Karnataka",
            "pincode": "560103",
        },
    ]
    if extra_addresses:
        addresses.extend(extra_addresses)
    num = customer_id.split("-")[-1]
    return {
        "customer_id":            customer_id,
        "name":                   f"Test Customer {num}",
        "email":                  f"customer{num}@test.com",
        "phone":                  f"+91-9{num}00000000",
        "tier":                   tier,
        "order_ids":              [f"ORD-7{num}00", f"ORD-7{num}01"],
        "preferred_refund_method": "HDFC_CREDIT",
        "addresses":              addresses,
    }


def _make_order(
    order_id:    str = "ORD-78321",
    customer_id: str = "CUST-001",
    status:      str = "processing",
    items:       list | None = None,
) -> dict[str, Any]:
    if items is None:
        items = [
            {
                "line_id": 1, "product_id": "PROD-LAPTOP-001",
                "name": "Dell Inspiron 15 Laptop",
                "quantity": 1, "unit_price": 55000.00, "status": "active",
            },
            {
                "line_id": 2, "product_id": "PROD-BAG-021",
                "name": "Laptop Backpack",
                "quantity": 1, "unit_price": 1500.00, "status": "active",
            },
            {
                "line_id": 3, "product_id": "PROD-MOUSE-005",
                "name": "Wireless Mouse",
                "quantity": 1, "unit_price": 800.00, "status": "active",
            },
        ]
    total = sum(
        i["unit_price"] * i["quantity"]
        for i in items
        if i["status"] == "active"
    )
    return {
        "order_id":    order_id,
        "customer_id": customer_id,
        "status":      status,
        "created_at":  "2025-05-01T10:30:00Z",
        "estimated_delivery": "2025-05-08",
        "tracking_number": "TRACK-7X9K2M" if status in ("shipped", "delivered") else None,
        "shipping_address": {
            "line1":   "12 MG Road",
            "line2":   "Koramangala",
            "city":    "Bengaluru",
            "state":   "Karnataka",
            "pincode": "560034",
        },
        "items":          items,
        "total_amount":   total,
        "payment_method": "HDFC_CREDIT",
    }


def _make_kb_articles() -> list[dict[str, Any]]:
    return [
        {
            "article_id": "KB-001",
            "title":      "Refund Policy and Threshold Rules",
            "tags":       ["refund", "threshold", "payments", "escalation", "return"],
            "content":    (
                "Acme Retail processes refunds automatically for amounts up to "
                "Rs.25,000. Above Rs.25,000 requires specialist escalation. "
                "24-hour response SLA."
            ),
            "last_updated": "2025-04-01",
            "applies_to":   ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-002",
            "title":      "Return Window Policy",
            "tags":       ["return", "window", "policy", "days"],
            "content":    "30-day return window. Electronics 7 days.",
            "last_updated": "2025-03-15",
            "applies_to":   ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-003",
            "title":      "Escalation SLA",
            "tags":       ["escalation", "sla", "specialist", "response"],
            "content":    "Specialist responds within 24 hours.",
            "last_updated": "2025-04-10",
            "applies_to":   ["electronics", "apparel", "home_goods"],
        },
        {
            "article_id": "KB-004",
            "title":      "Partial Cancellation and Address Update Rules",
            "tags":       ["cancel", "cancellation", "partial", "reship", "address"],
            "content":    (
                "Cancel items in placed/processing orders only. "
                "Address updates allowed before shipping."
            ),
            "last_updated": "2025-04-05",
            "applies_to":   ["electronics", "apparel", "home_goods"],
        },
    ]


def _make_payment_config() -> dict[str, Any]:
    return {
        "auto_refund_limit_inr": 25000,
        "supported_methods":     ["HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"],
        "refund_sla_days":       5,
        "behaviour": {
            "failure_rate":    0.0,   # no failures in tests
            "failure_code":    "504",
            "failure_message": "PAYMENT_GATEWAY_TIMEOUT",
        },
    }


# ---------------------------------------------------------------------------
# LLM mock helpers
# ---------------------------------------------------------------------------

def _mock_plan_response(intent: str, steps: list) -> MagicMock:
    """
    Build a fake completion object whose .choices[0].message.content
    is a valid plan JSON string.

    Used only so tests can call plan_completion.choices[0].message.content
    to extract the plan text when building mocks.
    """
    plan_json = json.dumps({"intent": intent, "steps": steps})
    mock_msg  = MagicMock()
    mock_msg.content = plan_json
    mock_choice  = MagicMock()
    mock_choice.message = mock_msg
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


def make_llm_mock(
    plan_completion: MagicMock,
    response_text:   str = "Your request has been processed.",
) -> AsyncMock:
    """
    Build an AsyncMock that correctly mimics Planner._call_llm and
    ResponseBuilder._call_llm.

    Both methods return a raw string (NOT a completion object):
      - First call  → plan JSON string  (consumed by Planner)
      - Second call → response text string (consumed by ResponseBuilder)

    The mock is set up with side_effect=[plan_str, response_str] so
    the first await returns the plan and the second returns the response.

    Usage in tests:
        mock = make_llm_mock(J1_PLAN, "Your order is processing.")
        with patch("agent.planner.Planner._call_llm", new=mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
                body = _post(client, "Where is ORD-78321?")
    """
    plan_str = plan_completion.choices[0].message.content
    return AsyncMock(side_effect=[plan_str, response_text])


# ---------------------------------------------------------------------------
# Pre-built plan mocks for J1, J2, J3
# ---------------------------------------------------------------------------

J1_PLAN = _mock_plan_response(
    "order_tracking",
    [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
)

J2_PLAN = _mock_plan_response(
    "compound",
    [
        {
            "action":     "cancel_item",
            "params":     {"order_id": "ORD-78321", "line_id": 2},
            "depends_on": [],
        },
        {
            "action":     "process_refund",
            "params":     {"order_id": "ORD-78321", "amount_inr": 1500.0, "method": "HDFC_CREDIT"},
            "depends_on": [0],
        },
        {
            "action":     "update_address",
            "params":     {"order_id": "ORD-78321", "address_label": "office"},
            "depends_on": [],
        },
    ],
)

J3_PLAN = _mock_plan_response(
    "escalation",
    [
        {
            "action": "escalate",
            "params": {
                "order_id":   "ORD-78500",
                "reason":     "Customer requesting full refund for damaged laptop. Exceeds threshold.",
                "amount_inr": 42000.0,
            },
            "depends_on": [],
        }
    ],
)


# ---------------------------------------------------------------------------
# Data directory fixture — isolated per test via tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """
    Create an isolated data directory with seeded JSON files.
    Every test gets a fresh tmp_path — no cross-test data contamination.
    """
    d = tmp_path / "data"
    d.mkdir()

    # CRM — customers and cases
    crm_data = {
        "customers": [
            _make_customer("CUST-001", tier="gold"),
            _make_customer("CUST-002", tier="platinum"),
            # CUST-003 has home address only (no office) — tests missing address
            {
                **_make_customer("CUST-003", tier="standard"),
                "addresses": [
                    {
                        "label":   "home",
                        "line1":   "22 Anna Salai",
                        "city":    "Chennai",
                        "state":   "Tamil Nadu",
                        "pincode": "600018",
                    }
                ],
            },
        ],
        "cases": [],
    }
    (d / "crm_cases.json").write_text(json.dumps(crm_data, indent=2), encoding="utf-8")

    # Orders — covers all statuses and boundary amounts
    orders_data = {
        "orders": [
            _make_order("ORD-78321", "CUST-001", "processing"),
            _make_order("ORD-78322", "CUST-001", "shipped"),
            _make_order("ORD-78323", "CUST-001", "delivered"),
            _make_order("ORD-78324", "CUST-001", "placed"),
            _make_order("ORD-78325", "CUST-001", "cancelled"),
            # Refund boundary: Rs.24,999 — just below threshold
            _make_order("ORD-78400", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "P-A", "name": "Item A",
                 "quantity": 1, "unit_price": 24999.0, "status": "active"},
            ]),
            # Refund boundary: Rs.25,000 — exactly at threshold
            _make_order("ORD-78401", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "P-B", "name": "Item B",
                 "quantity": 1, "unit_price": 25000.0, "status": "active"},
            ]),
            # Refund boundary: Rs.25,001 — just above threshold
            _make_order("ORD-78402", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "P-C", "name": "Item C",
                 "quantity": 1, "unit_price": 25001.0, "status": "active"},
            ]),
            # J3: Rs.42,000 high-value damaged item
            _make_order("ORD-78500", "CUST-001", "delivered", items=[
                {"line_id": 1, "product_id": "P-HV", "name": "Gaming Laptop",
                 "quantity": 1, "unit_price": 42000.0, "status": "active"},
            ]),
            # CUST-002 order — for cross-customer security tests
            _make_order("ORD-99001", "CUST-002", "processing"),
        ]
    }
    (d / "orders.json").write_text(json.dumps(orders_data, indent=2), encoding="utf-8")

    # KB articles
    (d / "kb_articles.json").write_text(
        json.dumps({"articles": _make_kb_articles()}, indent=2),
        encoding="utf-8",
    )

    # Payment config — failure_rate=0.0 so tests never hit simulated timeout
    (d / "payment_config.json").write_text(
        json.dumps(_make_payment_config(), indent=2),
        encoding="utf-8",
    )

    # Empty refunds — populated at runtime
    (d / "refunds.json").write_text(
        json.dumps({"refunds": []}, indent=2),
        encoding="utf-8",
    )

    # Sessions
    (d / "sessions.json").write_text(
        json.dumps({
            "sessions": [
                {"session_id": "sess-cust001", "customer_id": "CUST-001"},
                {"session_id": "sess-cust002", "customer_id": "CUST-002"},
                {"session_id": "sess-cust003", "customer_id": "CUST-003"},
            ]
        }, indent=2),
        encoding="utf-8",
    )

    # Users (auth credentials — separate from CRM customer data)
    users_data = {
        "users": [
            _make_user("alice", "CUST-001", password="Atlas@123"),
            _make_user("bob",   "CUST-002", password="Atlas@456"),
        ]
    }
    (d / "users.json").write_text(
        json.dumps(users_data, indent=2), encoding="utf-8"
    )

    return d


# ---------------------------------------------------------------------------
# Environment patcher — points all repos to the isolated data_dir
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_env(data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ORDERS_DATA_PATH",    str(data_dir / "orders.json"))
    monkeypatch.setenv("CRM_DATA_PATH",       str(data_dir / "crm_cases.json"))
    monkeypatch.setenv("KB_DATA_PATH",        str(data_dir / "kb_articles.json"))
    monkeypatch.setenv("PAYMENT_CONFIG_PATH", str(data_dir / "payment_config.json"))
    monkeypatch.setenv("REFUNDS_DATA_PATH",   str(data_dir / "refunds.json"))
    monkeypatch.setenv("SESSIONS_DATA_PATH",  str(data_dir / "sessions.json"))
    monkeypatch.setenv("USERS_DATA_PATH",     str(data_dir / "users.json"))


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def client(patched_env) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient with:
      - Isolated data directory (fresh per test)
      - App reloaded so lifespan startup picks up patched env vars
      - raise_server_exceptions=False so 422/500 responses don't raise
        in the test — we assert on status codes instead
    """
    import importlib
    import main as main_module
    # Clear session history between tests to prevent cross-test contamination
    from agent.orchestrator import _session_history
    _session_history.clear()

    importlib.reload(main_module)

    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Async tool fixtures
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