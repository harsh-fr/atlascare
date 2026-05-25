"""
tests/test_contracts.py
========================
API contract and schema correctness tests.
"""

import re
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from tests.conftest import _mock_plan_response, make_llm_mock


# ---------------------------------------------------------------------------
# Helper — post a valid query with mocked LLM
# ---------------------------------------------------------------------------

def _valid_query(client, message="Where is my order ORD-78321?"):
    plan = _mock_plan_response("order_tracking",
        [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}])
    mock = make_llm_mock(plan, "Your order is being processed.")
    with patch("agent.planner.Planner._call_llm", new=mock):
        with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
            resp = client.post("/query",
                json={"message": message, "session_id": "sess-cust001"})
    assert resp.status_code == 200
    return resp.json()


# ===========================================================================
# GET /health
# ===========================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_health_returns_status_ok(self, client):
        assert client.get("/health").json().get("status") == "ok"

    def test_health_returns_json(self, client):
        assert "application/json" in client.get("/health").headers.get("content-type", "")


# ===========================================================================
# POST /query — request contract
# ===========================================================================

class TestQueryRequestContract:

    def test_valid_request_returns_200(self, client):
        assert _valid_query(client) is not None

    def test_missing_message_returns_422(self, client):
        assert client.post("/query", json={"session_id": "sess-cust001"}).status_code == 422

    def test_missing_session_id_returns_422(self, client):
        assert client.post("/query", json={"message": "Hello"}).status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post("/query", json={}).status_code == 422

    def test_null_message_returns_422(self, client):
        assert client.post("/query", json={"message": None, "session_id": "sess-cust001"}).status_code == 422

    def test_null_session_id_returns_422(self, client):
        assert client.post("/query", json={"message": "Hello", "session_id": None}).status_code == 422

    def test_whitespace_message_returns_422(self, client):
        resp = client.post("/query", json={"message": "   ", "session_id": "sess-cust001"})
        assert resp.status_code == 422

    def test_422_response_is_json_serialisable(self, client):
        """422 error body must be valid JSON — no ValueError objects in detail."""
        resp = client.post("/query", json={"message": "   ", "session_id": "sess-cust001"})
        assert resp.status_code == 422
        # Must not raise — if ValueError leaked, json() would fail
        body = resp.json()
        assert "detail" in body
        # Re-serialise to confirm it's clean
        json.dumps(body)   # raises TypeError if non-serialisable values present

    def test_injection_session_id_returns_422(self, client):
        resp = client.post("/query",
            json={"message": "Hello", "session_id": "sess'; DROP TABLE --"})
        assert resp.status_code == 422

    def test_extra_fields_ignored(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}])
        mock = make_llm_mock(plan, "Your order.")
        with patch("agent.planner.Planner._call_llm", new=mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
                resp = client.post("/query", json={
                    "message":          "Where is ORD-78321?",
                    "session_id":       "sess-cust001",
                    "unexpected_field": "should be ignored",
                })
        assert resp.status_code == 200


# ===========================================================================
# POST /query — response contract
# ===========================================================================

class TestQueryResponseContract:

    def test_response_has_response_field(self, client):
        assert "response" in _valid_query(client)

    def test_response_field_is_string(self, client):
        assert isinstance(_valid_query(client)["response"], str)

    def test_response_field_is_non_empty(self, client):
        assert len(_valid_query(client)["response"]) > 0

    def test_trace_field_present(self, client):
        assert "trace" in _valid_query(client)

    def test_trace_has_trace_id(self, client):
        assert "trace_id" in _valid_query(client)["trace"]

    def test_trace_has_session_id(self, client):
        assert "session_id" in _valid_query(client)["trace"]

    def test_trace_has_latency_ms(self, client):
        assert "latency_ms" in _valid_query(client)["trace"]

    def test_trace_has_tool_calls(self, client):
        assert "tool_calls" in _valid_query(client)["trace"]

    def test_trace_id_starts_with_trc(self, client):
        assert _valid_query(client)["trace"]["trace_id"].startswith("trc-")

    def test_trace_id_is_string(self, client):
        assert isinstance(_valid_query(client)["trace"]["trace_id"], str)

    def test_session_id_echoed_in_trace(self, client):
        assert _valid_query(client)["trace"]["session_id"] == "sess-cust001"

    def test_latency_ms_is_integer(self, client):
        assert isinstance(_valid_query(client)["trace"]["latency_ms"], int)

    def test_latency_ms_is_positive(self, client):
        assert _valid_query(client)["trace"]["latency_ms"] > 0

    def test_tool_calls_is_list(self, client):
        assert isinstance(_valid_query(client)["trace"]["tool_calls"], list)

    def test_tool_calls_never_null(self, client):
        assert _valid_query(client)["trace"]["tool_calls"] is not None

    def test_tool_call_has_required_fields(self, client):
        body = _valid_query(client)
        for tc in body["trace"]["tool_calls"]:
            assert "tool"       in tc
            assert "action"     in tc
            assert "status"     in tc
            assert "latency_ms" in tc

    def test_tool_call_tool_is_string(self, client):
        for tc in _valid_query(client)["trace"]["tool_calls"]:
            assert isinstance(tc["tool"], str)

    def test_tool_call_status_is_string(self, client):
        for tc in _valid_query(client)["trace"]["tool_calls"]:
            assert isinstance(tc["status"], str)

    def test_no_extra_top_level_fields(self, client):
        body  = _valid_query(client)
        extra = set(body.keys()) - {"response", "trace"}
        assert not extra, f"Unexpected top-level keys: {extra}"

    def test_response_is_valid_json_serialisable(self, client):
        body = _valid_query(client)
        serialised = json.dumps(body)
        reparsed   = json.loads(serialised)
        assert reparsed["trace"]["trace_id"] == body["trace"]["trace_id"]


# ===========================================================================
# Data schema conformance
# ===========================================================================

class TestDataSchemaConformance:

    def test_orders_schema_order_id_pattern(self, patched_env):
        from repositories.order_repository import OrderRepository
        pattern = re.compile(r"^ORD-\d{5}$")
        for order in OrderRepository().list_all():
            assert pattern.match(order["order_id"]), \
                f"order_id '{order['order_id']}' does not match schema."

    def test_orders_schema_total_equals_sum_of_active_items(self, patched_env):
        from repositories.order_repository import OrderRepository
        for order in OrderRepository().list_all():
            expected = sum(
                i["unit_price"] * i["quantity"]
                for i in order["items"] if i["status"] == "active"
            )
            assert abs(order["total_amount"] - expected) < 0.01, \
                f"Order {order['order_id']}: total={order['total_amount']} expected={expected}"

    def test_orders_schema_status_valid(self, patched_env):
        from repositories.order_repository import OrderRepository
        valid = {"placed", "processing", "shipped", "delivered", "cancelled"}
        for order in OrderRepository().list_all():
            assert order["status"] in valid

    def test_crm_schema_customer_id_pattern(self, patched_env):
        from repositories.crm_repository import CrmRepository
        pattern = re.compile(r"^CUST-\d{3}$")
        for c in CrmRepository().list_all_customers():
            assert pattern.match(c["customer_id"])

    def test_crm_schema_tier_valid(self, patched_env):
        from repositories.crm_repository import CrmRepository
        valid = {"standard", "silver", "gold", "platinum"}
        for c in CrmRepository().list_all_customers():
            assert c["tier"] in valid

    def test_kb_articles_have_required_fields(self, patched_env):
        from repositories.kb_repository import KbRepository
        required = {"article_id", "title", "tags", "content", "last_updated"}
        for a in KbRepository().get_all_articles():
            missing = required - set(a.keys())
            assert not missing, f"KB article {a.get('article_id')} missing: {missing}"

    def test_kb_article_id_pattern(self, patched_env):
        from repositories.kb_repository import KbRepository
        pattern = re.compile(r"^KB-\d{3}$")
        for a in KbRepository().get_all_articles():
            assert pattern.match(a["article_id"])

    def test_payment_config_threshold_is_25000(self, patched_env):
        from repositories.payment_repository import PaymentRepository
        assert PaymentRepository().get_auto_refund_limit() == 25000.0

    def test_payment_config_has_all_supported_methods(self, patched_env):
        from repositories.payment_repository import PaymentRepository
        methods  = set(PaymentRepository().get_supported_methods())
        required = {"HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"}
        assert required.issubset(methods)

    def test_refund_record_schema_after_creation(self, patched_env):
        from services.refund_service import RefundService
        record   = RefundService().create_refund_record(
            "ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001", 5)
        required = {"refund_id", "order_id", "customer_id", "amount_inr",
                    "method", "status", "sla_days", "created_at"}
        assert required.issubset(set(record.keys()))
        assert re.match(r"^REF-\d{5}-[A-F0-9]{8}$", record["refund_id"])
        assert record["status"]     == "initiated"
        assert record["amount_inr"] == 1500.0

    def test_case_schema_after_creation(self, patched_env):
        from services.escalation_service import EscalationService
        case = EscalationService().create_case(
            "CUST-001", "ORD-78321", "Test escalation.",
            42000.0, "trc-test123", "high")
        required = {"case_id", "customer_id", "order_id", "status",
                    "priority", "description", "created_at", "trace_id"}
        assert required.issubset(set(case.keys()))
        assert re.match(r"^CASE-[A-Z0-9]{6}$", case["case_id"])
        assert case["status"]   == "open"
        assert case["priority"] == "high"
        assert case["trace_id"] == "trc-test123"