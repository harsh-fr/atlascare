# AtlasCare — Test Plan

## 1. Testing Philosophy

The test suite is designed assuming evaluators will intentionally attempt to break the system. Tests are not limited to happy paths. Every boundary condition, failure mode, and security concern identified in the requirements has a corresponding test.

All tests are deterministic — LLM calls are mocked with fixed JSON plans. Tests never require a live Gemini API key.

---

## 2. Test Coverage Summary

### 2.1 Journey Tests (`test_journeys.py`) — 18 tests

| Test | What is verified |
|------|-----------------|
| J1: returns 200 | Basic endpoint liveness |
| J1: latency < 3s | SLA enforcement (wall-clock, not trace) |
| J1: exactly 1 OMS call | Deterministic tool dispatch |
| J1: no CRM/payment calls | Tool isolation |
| J1: no escalation | Correct intent routing |
| J1: trace fields present | API contract compliance |
| J1: shipped order tracking | Real data returned |
| J2: all 3 steps in trace | Full compound execution |
| J2: cancel success | Processing order cancellable |
| J2: refund success | Sub-threshold refund processed |
| J2: address update recorded | Address resolution works |
| J2: failed step recorded | Audit trail completeness |
| J2: dependent step skipped | Dependency resolution logic |
| J3: payment never called | CRITICAL threshold enforcement |
| J3: CRM case created | Escalation persistence |
| J3: case has trace_id | Audit linkage |
| J3: structured handoff | Handoff summary format |
| J3: priority=high | Escalation case priority |
| J3: polite response | User-facing message quality |
| J3: pre-guardrail blocks | GR-001 fires before LLM |
| J3: case_id schema | Pattern conformance |

### 2.2 Guardrail Tests (`test_guardrails.py`) — 45 tests

- `_extract_amounts`: all currency formats (₹, Rs., INR, rupees, comma-separated)
- GR-001: boundary values (24999/25000/25001), no-keyword bypass, multi-amount
- GR-001: polite message, no internal details leaked, tracer records trigger
- GR-002: empty string, whitespace-only
- GR-003: exactly 2000 chars (allow), 2001 chars (block)
- Rule priority: GR-002 fires before GR-001 on empty message
- GR-004: all four combinations of payment+escalation
- PaymentTool: independently blocks ₹25,001, allows ₹25,000, rejects zero, rejects bad method
- HTTP: blocked request returns 200 with trace, non-empty response
- AQ: ambiguous query with no conversation history → blocked, prompts for order ID
- AQ: ambiguous follow-up ("what about the delivery date?") after user provided ORD-XXXXX in prior turn → not blocked, passes through to tool agent
- AQ: ambiguous follow-up where order ID appears only in a tool result message → not blocked

### 2.3 Security Tests (`test_security.py`) — 35 tests

- Cross-customer access via all 4 tool paths (get_order, cancel_item, process_refund, update_address)
- Vague error messages — no owner ID disclosed
- Order enumeration prevention
- Unknown, empty, missing session_id handling
- 7 parametrized injection patterns (SQL, path traversal, XSS, whitespace, quotes)
- Unit-level ownership assertion: correct customer passes, wrong customer raises, error message safe

### 2.4 Contract Tests (`test_contracts.py`) — 48 tests

- GET /health: 200, `{"status":"ok"}`, JSON content-type
- POST /query request: missing/null/whitespace fields → 422, extra fields ignored
- POST /query response: all required fields, correct types, `trc-` prefix, session echoed, latency > 0, tool_calls never null, no extra top-level keys, full JSON round-trip
- Data schema: order_id regex, total = sum of active items, status enums, customer tiers, KB fields, auto_refund_limit = 25000, refund/case record schema after creation

### 2.5 Edge Case Tests (`test_edge_cases.py`) — 47 tests

- Non-existent order: recorded as error, returns 200
- Non-existent line_id: error in trace
- Already-cancelled item: error in trace
- Shipped/delivered order cancellation: error in trace
- Missing office address: error in trace
- KB empty/unmatched tags: returns []
- Gateway retry: fail-then-succeed, all retries exhausted
- Planner: malformed JSON, unknown intent, LLM timeout → all return safe 200
- Hallucination: no fabricated tracking numbers, no invented order details
- Trace: unique trace_id per request, latency always present, failed steps visible, ordering correct
- Validator unit tests: all 11 validators

---

## 3. Deliberately Out of Scope (This Submission)

| Item | Reason |
|------|--------|
| Live Gemini API integration tests | Require live API key; would be flaky in CI |
| Load / stress tests | Require infrastructure; 18K/day scale needs k6 or Locust setup |
| Multi-process concurrency tests | JSON store is single-process by design; real DB needed for this |
| Authentication / JWT tests | Spec explicitly says no auth system — session store is simulation |
| Webhook / async notification tests | Not in spec scope |
| Admin / article management tests | No write path for KB at runtime |

---

## 4. Before Going Live — Additional Tests Needed

1. **Live LLM integration tests**: end-to-end with real Gemini API key, verifying actual plan JSON for the three journeys
2. **Load tests**: 18K requests/day baseline, spike to 3x, p99 latency under 3s
3. **LLM prompt regression suite**: capture real LLM outputs, detect prompt drift on model updates
4. **Chaos tests**: kill the LLM mid-request, corrupt JSON files, simulate disk full
5. **Database migration tests**: when JSON is replaced with PostgreSQL, verify repository contracts hold
6. **Multi-tenant isolation tests**: verify session store scales to millions of sessions
7. **Penetration testing**: third-party security review of the session simulation