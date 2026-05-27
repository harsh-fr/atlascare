"""
tests/conftest.py
==================
Pytest fixtures and shared test configuration for AtlasCare.

Mock strategy (new LangGraph architecture)
------------------------------------------
All LLM calls go through agent.graph._groq_client (lazy singleton).
Tests patch this with a MagicMock whose chat.completions.create is an
AsyncMock with side_effect=<list of completion mocks>.

Call order for a typical tool-using request (2 LLM calls):
  1. tool_agent_node (70B) → make_tool_mock(...)  # decide what tools to call
  2. tool_executor_node    → real tools run (no LLM call)
  3. responder_node  (8B)  → make_text_mock(...)  # generate customer reply

For no-tool requests (greetings, KB queries — 1 LLM call):
  1. tool_agent_node (70B) → make_done_mock("text")  # direct reply, used as-is

For escalation (1 LLM call — deterministic response, no responder call):
  1. tool_agent_node (70B) → make_tool_mock("escalate", ...)
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
os.environ.setdefault("GROQ_API_KEY",               "test-key-not-real")
os.environ.setdefault("GROQ_BASE_URL",              "https://api.groq.com/openai/v1")
os.environ.setdefault("PLANNER_MODEL",              "llama-3.3-70b-versatile")
os.environ.setdefault("RESPONSE_MODEL",             "llama-3.1-8b-instant")
os.environ.setdefault("AUTO_REFUND_LIMIT_INR",      "25000.0")
os.environ.setdefault("PAYMENT_MAX_RETRIES",        "3")
os.environ.setdefault("PAYMENT_RETRY_BASE_DELAY_S", "0.0")


# ---------------------------------------------------------------------------
# LLM completion mock helpers
# ---------------------------------------------------------------------------

def make_tool_mock(tool_name: str, args: dict) -> MagicMock:
    """70B completion that requests a single tool call."""
    tc = MagicMock()
    tc.id = f"call_{tool_name}_01"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def make_multi_tool_mock(tool_calls: list[tuple[str, dict]]) -> MagicMock:
    """70B completion that requests multiple tool calls in parallel."""
    tcs = []
    for i, (tool_name, args) in enumerate(tool_calls):
        tc = MagicMock()
        tc.id = f"call_{tool_name}_{i:02d}"
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(args)
        tcs.append(tc)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = tcs

    choice = MagicMock()
    choice.message = msg

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def make_done_mock(text: str = "") -> MagicMock:
    """70B completion with no tool calls — signals the agent is done planning."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg

    completion = MagicMock()
    completion.choices = [choice]
    return completion


def make_text_mock(
    text: str = "Your request has been processed. Is there anything else I can help with?",
) -> MagicMock:
    """8B completion with a customer-facing text response."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None

    choice = MagicMock()
    choice.message = msg

    completion = MagicMock()
    completion.choices = [choice]
    return completion


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
            "failure_rate":    0.0,
            "failure_code":    "504",
            "failure_message": "PAYMENT_GATEWAY_TIMEOUT",
        },
    }


# ---------------------------------------------------------------------------
# Data directory fixture — isolated per test via tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()

    crm_data = {
        "customers": [
            _make_customer("CUST-001", tier="gold"),
            _make_customer("CUST-002", tier="platinum"),
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

    orders_data = {
        "orders": [
            _make_order("ORD-78321", "CUST-001", "processing"),
            _make_order("ORD-78322", "CUST-001", "shipped"),
            _make_order("ORD-78323", "CUST-001", "delivered"),
            _make_order("ORD-78324", "CUST-001", "placed"),
            _make_order("ORD-78325", "CUST-001", "cancelled"),
            _make_order("ORD-78400", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "P-A", "name": "Item A",
                 "quantity": 1, "unit_price": 24999.0, "status": "active"},
            ]),
            _make_order("ORD-78401", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "P-B", "name": "Item B",
                 "quantity": 1, "unit_price": 25000.0, "status": "active"},
            ]),
            _make_order("ORD-78402", "CUST-001", "processing", items=[
                {"line_id": 1, "product_id": "P-C", "name": "Item C",
                 "quantity": 1, "unit_price": 25001.0, "status": "active"},
            ]),
            _make_order("ORD-78500", "CUST-001", "delivered", items=[
                {"line_id": 1, "product_id": "P-HV", "name": "Gaming Laptop",
                 "quantity": 1, "unit_price": 42000.0, "status": "active"},
            ]),
            _make_order("ORD-99001", "CUST-002", "processing"),
        ]
    }
    (d / "orders.json").write_text(json.dumps(orders_data, indent=2), encoding="utf-8")
    (d / "kb_articles.json").write_text(
        json.dumps({"articles": _make_kb_articles()}, indent=2), encoding="utf-8"
    )
    (d / "payment_config.json").write_text(
        json.dumps(_make_payment_config(), indent=2), encoding="utf-8"
    )
    (d / "refunds.json").write_text(
        json.dumps({"refunds": []}, indent=2), encoding="utf-8"
    )
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
    (d / "users.json").write_text(
        json.dumps({"users": [
            _make_user("alice", "CUST-001", password="Atlas@123"),
            _make_user("bob",   "CUST-002", password="Atlas@456"),
        ]}, indent=2),
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Environment patcher
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
    import importlib
    import agent.graph as graph_module
    import main as main_module

    # Reload graph so module-level tool singletons (_oms, _crm, etc.)
    # are re-created using the patched env data paths for this test.
    importlib.reload(graph_module)
    graph_module._groq_client = None  # reset lazy singleton
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
