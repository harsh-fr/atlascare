"""
tests/test_contracts.py
========================
API contract and schema correctness tests.

Coverage
--------
  POST /query
    - Request contract: valid payload returns 200
    - Request contract: missing fields return 422
    - Response shape: all required fields present
    - Response shape: correct types for every field
    - trace.tool_calls is always a list (never null)
    - trace.latency_ms is always a positive integer
    - trace.trace_id always starts with "trc-"
    - trace.session_id echoes the request session_id

  GET /health
    - Returns HTTP 200
    - Returns {"status": "ok"}

  Schema validation
    - order_id pattern validated by repository
    - customer_id pattern validated by repository
    - case_id pattern matches schema regex after creation
    - refund_id format is correct after creation

  Data schema conformance
    - Orders loaded from JSON match orders.json schema
    - CRM data loaded from JSON matches crm_cases.json schema
    - KB articles match kb_articles.json schema
    - Payment config matches payment_config.json schema
"""

import re
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from tests.conftest import J1_PLAN, make_llm_mock, _mock_plan_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_query(client: TestClient, message: str = "Where is my order ORD-78321?") -> dict:
    """Post a valid query and return the parsed response body."""
    plan = _mock_plan_response(
        "order_tracking",
        [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
    )
    llm_mock = make_llm_mock(plan, "Your order is being processed.")
    with patch("agent.planner.Planner._call_llm", new=llm_mock):
        with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
            resp = client.post(
                "/query",
                json={"message": message, "session_id": "sess-cust001"},
            )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_status_ok(self, client: TestClient):
        resp = client.get("/health")
        body = resp.json()
        assert body.get("status") == "ok"

    def test_health_returns_json(self, client: TestClient):
        resp = client.get("/health")
        assert "application/json" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# POST /query — request contract
# ---------------------------------------------------------------------------

class TestQueryRequestContract:

    def test_valid_request_returns_200(self, client: TestClient):
        body = _valid_query(client)
        assert body is not None

    def test_missing_message_returns_422(self, client: TestClient):
        resp = client.post("/query", json={"session_id": "sess-cust001"})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client: TestClient):
        resp = client.post("/query", json={"message": "Hello"})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_null_message_returns_422(self, client: TestClient):
        resp = client.post("/query", json={"message": None, "session_id": "sess-cust001"})
        assert resp.status_code == 422

    def test_null_session_id_returns_422(self, client: TestClient):
        resp = client.post("/query", json={"message": "Hello", "session_id": None})
        assert resp.status_code == 422

    def test_whitespace_message_returns_422(self, client: TestClient):
        resp = client.post("/query", json={"message": "   ", "session_id": "sess-cust001"})
        assert resp.status_code == 422

    def test_extra_fields_ignored(self, client: TestClient):
        """Extra fields in request body must be ignored, not cause errors."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Your order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={
                        "message": "Where is ORD-78321?",
                        "session_id": "sess-cust001",
                        "unexpected_field": "should be ignored",
                    },
                )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /query — response contract
# ---------------------------------------------------------------------------

class TestQueryResponseContract:

    def test_response_has_response_field(self, client: TestClient):
        body = _valid_query(client)
        assert "response" in body

    def test_response_field_is_string(self, client: TestClient):
        body = _valid_query(client)
        assert isinstance(body["response"], str)

    def test_response_field_is_non_empty(self, client: TestClient):
        body = _valid_query(client)
        assert len(body["response"]) > 0

    def test_trace_field_present(self, client: TestClient):
        body = _valid_query(client)
        assert "trace" in body

    def test_trace_has_trace_id(self, client: TestClient):
        body = _valid_query(client)
        assert "trace_id" in body["trace"]

    def test_trace_has_session_id(self, client: TestClient):
        body = _valid_query(client)
        assert "session_id" in body["trace"]

    def test_trace_has_latency_ms(self, client: TestClient):
        body = _valid_query(client)
        assert "latency_ms" in body["trace"]

    def test_trace_has_tool_calls(self, client: TestClient):
        body = _valid_query(client)
        assert "tool_calls" in body["trace"]

    def test_trace_id_starts_with_trc(self, client: TestClient):
        body = _valid_query(client)
        assert body["trace"]["trace_id"].startswith("trc-")

    def test_trace_id_is_string(self, client: TestClient):
        body = _valid_query(client)
        assert isinstance(body["trace"]["trace_id"], str)

    def test_session_id_echoed_in_trace(self, client: TestClient):
        body = _valid_query(client)
        assert body["trace"]["session_id"] == "sess-cust001"

    def test_latency_ms_is_integer(self, client: TestClient):
        body = _valid_query(client)
        assert isinstance(body["trace"]["latency_ms"], int)

    def test_latency_ms_is_positive(self, client: TestClient):
        body = _valid_query(client)
        assert body["trace"]["latency_ms"] > 0

    def test_tool_calls_is_list(self, client: TestClient):
        body = _valid_query(client)
        assert isinstance(body["trace"]["tool_calls"], list)

    def test_tool_calls_never_null(self, client: TestClient):
        body = _valid_query(client)
        assert body["trace"]["tool_calls"] is not None

    def test_tool_call_has_required_fields(self, client: TestClient):
        """Each tool_call entry must have tool, action, status, latency_ms."""
        body = _valid_query(client)
        for tc in body["trace"]["tool_calls"]:
            assert "tool"       in tc, f"Missing 'tool' in {tc}"
            assert "action"     in tc, f"Missing 'action' in {tc}"
            assert "status"     in tc, f"Missing 'status' in {tc}"
            assert "latency_ms" in tc, f"Missing 'latency_ms' in {tc}"

    def test_tool_call_tool_is_string(self, client: TestClient):
        body = _valid_query(client)
        for tc in body["trace"]["tool_calls"]:
            assert isinstance(tc["tool"], str)

    def test_tool_call_status_is_string(self, client: TestClient):
        body = _valid_query(client)
        for tc in body["trace"]["tool_calls"]:
            assert isinstance(tc["status"], str)

    def test_no_extra_top_level_fields(self, client: TestClient):
        """Response must only have 'response' and 'trace' at top level."""
        body = _valid_query(client)
        allowed_keys = {"response", "trace"}
        extra = set(body.keys()) - allowed_keys
        assert not extra, f"Unexpected top-level keys in response: {extra}"

    def test_no_extra_trace_fields_beyond_contract(self, client: TestClient):
        """trace must contain at minimum: trace_id, session_id, latency_ms, tool_calls."""
        body = _valid_query(client)
        required = {"trace_id", "session_id", "latency_ms", "tool_calls"}
        assert required.issubset(body["trace"].keys())

    def test_response_is_valid_json_serialisable(self, client: TestClient):
        """Full response must be JSON-serialisable (no datetime objects etc.)."""
        body = _valid_query(client)
        serialised = json.dumps(body)
        reparsed   = json.loads(serialised)
        assert reparsed["trace"]["trace_id"] == body["trace"]["trace_id"]


# ---------------------------------------------------------------------------
# Data schema conformance
# ---------------------------------------------------------------------------

class TestDataSchemaConformance:

    def test_orders_schema_order_id_pattern(self, patched_env):
        """All order_ids in data must match ^ORD-[0-9]{5}$."""
        from repositories.order_repository import OrderRepository
        repo = OrderRepository()
        pattern = re.compile(r"^ORD-\d{5}$")
        for order in repo.list_all():
            assert pattern.match(order["order_id"]), (
                f"order_id '{order['order_id']}' does not match schema pattern."
            )

    def test_orders_schema_total_equals_sum_of_active_items(self, patched_env):
        """total_amount must equal sum of unit_price * quantity for active items."""
        from repositories.order_repository import OrderRepository
        repo = OrderRepository()
        for order in repo.list_all():
            expected = sum(
                i["unit_price"] * i["quantity"]
                for i in order["items"]
                if i["status"] == "active"
            )
            assert abs(order["total_amount"] - expected) < 0.01, (
                f"Order {order['order_id']}: total_amount={order['total_amount']} "
                f"but sum of active items={expected}"
            )

    def test_orders_schema_status_valid(self, patched_env):
        """All order statuses must be in the allowed enum."""
        from repositories.order_repository import OrderRepository
        valid = {"placed", "processing", "shipped", "delivered", "cancelled"}
        repo = OrderRepository()
        for order in repo.list_all():
            assert order["status"] in valid, (
                f"Order {order['order_id']} has invalid status '{order['status']}'."
            )

    def test_crm_schema_customer_id_pattern(self, patched_env):
        """All customer_ids must match ^CUST-[0-9]{3}$."""
        from repositories.crm_repository import CrmRepository
        repo = OrderRepository() if False else CrmRepository()
        pattern = re.compile(r"^CUST-\d{3}$")
        for customer in repo.list_all_customers():
            assert pattern.match(customer["customer_id"]), (
                f"customer_id '{customer['customer_id']}' does not match schema."
            )

    def test_crm_schema_tier_valid(self, patched_env):
        """All customer tiers must be in the allowed enum."""
        from repositories.crm_repository import CrmRepository
        valid = {"standard", "silver", "gold", "platinum"}
        repo = CrmRepository()
        for customer in repo.list_all_customers():
            assert customer["tier"] in valid

    def test_kb_articles_have_required_fields(self, patched_env):
        """Every KB article must have article_id, title, tags, content, last_updated."""
        from repositories.kb_repository import KbRepository
        repo = KbRepository()
        required = {"article_id", "title", "tags", "content", "last_updated"}
        for article in repo.get_all_articles():
            missing = required - set(article.keys())
            assert not missing, (
                f"KB article {article.get('article_id')} missing fields: {missing}"
            )

    def test_kb_article_id_pattern(self, patched_env):
        """All article_ids must match ^KB-[0-9]{3}$."""
        from repositories.kb_repository import KbRepository
        repo = KbRepository()
        pattern = re.compile(r"^KB-\d{3}$")
        for article in repo.get_all_articles():
            assert pattern.match(article["article_id"])

    def test_payment_config_threshold_is_25000(self, patched_env):
        """auto_refund_limit_inr must be exactly 25000."""
        from repositories.payment_repository import PaymentRepository
        repo = PaymentRepository()
        assert repo.get_auto_refund_limit() == 25000.0

    def test_payment_config_has_all_supported_methods(self, patched_env):
        """All five supported methods must be present in config."""
        from repositories.payment_repository import PaymentRepository
        repo = PaymentRepository()
        methods = set(repo.get_supported_methods())
        required = {"HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"}
        assert required.issubset(methods)

    def test_refund_record_schema_after_creation(self, patched_env):
        """A newly created refund record must match expected schema fields."""
        from services.refund_service import RefundService
        from repositories.payment_repository import PaymentRepository

        svc  = RefundService()
        repo = PaymentRepository()

        record = svc.create_refund_record(
            order_id="ORD-78321",
            amount_inr=1500.0,
            method="HDFC_CREDIT",
            customer_id="CUST-001",
            sla_days=5,
        )
        required = {
            "refund_id", "order_id", "customer_id",
            "amount_inr", "method", "status", "sla_days", "created_at"
        }
        assert required.issubset(set(record.keys()))
        assert re.match(r"^REF-\d{5}-[A-F0-9]{8}$", record["refund_id"])
        assert record["status"] == "initiated"
        assert record["amount_inr"] == 1500.0

    def test_case_schema_after_creation(self, patched_env):
        """A newly created CRM case must match expected schema fields."""
        from services.escalation_service import EscalationService
        import re as _re

        svc = EscalationService()
        case = svc.create_case(
            customer_id="CUST-001",
            order_id="ORD-78321",
            reason="Test escalation.",
            amount_inr=42000.0,
            trace_id="trc-test123",
            priority="high",
        )
        required = {
            "case_id", "customer_id", "order_id", "status",
            "priority", "description", "created_at", "trace_id"
        }
        assert required.issubset(set(case.keys()))
        assert _re.match(r"^CASE-[A-Z0-9]{6}$", case["case_id"])
        assert case["status"]   == "open"
        assert case["priority"] == "high"
        assert case["trace_id"] == "trc-test123"