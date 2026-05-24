# AtlasCare — Architecture Document

## 1. System Overview

AtlasCare is a production-grade Agentic AI layer for Acme Retail's customer support platform. It handles ~18,000 interactions/day autonomously for Tier-1 queries while safely escalating those it cannot resolve.

**Core design principle: Deterministic > Generative.**
The LLM is used only where intelligence is genuinely valuable (intent extraction, natural language phrasing). All policy enforcement, arithmetic, ownership checks, and business rules live in deterministic Python code.

---

## 2. Architecture Layers

```
HTTP Request (POST /query)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│  FastAPI endpoint — timing, session wiring, trace   │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│               agent/orchestrator.py                 │
│  Pipeline coordinator — wires all components        │
│  6-step flow: session → pre-guard → plan →          │
│               execute → post-guard → respond        │
└──┬────────────┬──────────────┬────────────┬─────────┘
   │            │              │            │
   ▼            ▼              ▼            ▼
planner.py  guardrails.py  executor.py  response_builder.py
(LLM)       (Code rules)   (Dispatch)   (LLM phrasing)
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
         oms_tool.py      crm_tool.py      payment_tool.py
         kb_tool.py
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
order_repo  crm_repo  payment_repo
kb_repo
    │
  data/*.json  (JSON-backed, atomic writes)
```

---

## 3. Request Pipeline (6 Steps)

| Step | Component | Who does it | What happens |
|------|-----------|-------------|--------------|
| 1 | SessionStore | Code | `session_id` → `customer_id` resolution |
| 2 | Guardrails.pre_check | Code | Policy rules before LLM sees message |
| 3 | Planner | LLM (Gemini) | Intent extraction → typed ActionPlan |
| 4 | Executor | Code | Tool dispatch with ownership validation |
| 5 | Guardrails.post_check | Code | Verify execution outcomes comply |
| 6 | ResponseBuilder | LLM (Gemini) | Ground verified data → natural reply |

---

## 4. Planning Strategy

The Planner sends the customer message to Gemini 2.5 Flash with a **constrained system prompt** that forces JSON output conforming to a strict action schema. The model returns an `ActionPlan` with:

- `intent`: classified intent (order_tracking, compound, escalation, etc.)
- `steps[]`: ordered `ActionStep` list with `action`, `params`, `depends_on`

**Temperature = 0** for maximum determinism. Output is validated in `_parse_and_validate()` before reaching the Executor. Malformed LLM output raises `PlannerError` — the pipeline never passes an unvalidated plan to tools.

---

## 5. Tool Design

Tools are MCP-inspired typed async interfaces. They are the **only** layer that touches repositories. Agent code never accesses JSON files directly.

| Tool | Key methods | Backed by |
|------|------------|-----------|
| OmsTool | get_order, cancel_item, update_shipping_address | OrderRepository |
| CrmTool | get_customer, create_case, get_cases | CrmRepository |
| PaymentTool | process_refund | PaymentRepository |
| KbTool | search, get_article | KbRepository |

Tools are swappable — replacing JSON repos with REST APIs requires only changing the repository layer.

---

## 6. Guardrails (Defence in Depth)

Three independent layers enforce the Rs.25,000 threshold:

| Layer | Where | When |
|-------|-------|------|
| GR-001 | Guardrails.pre_check | Before LLM — regex extracts amount from message |
| PaymentTool._enforce_threshold | payment_tool.py | At call time — Decimal comparison |
| GR-004 | Guardrails.post_check | After execution — verifies no payment on escalation case |

**Rule inventory:**
- GR-001: High-value refund → block + route to escalation
- GR-002: Empty message → reject
- GR-003: Message > 2,000 chars → reject (prompt injection defence)
- GR-004: Payment success + escalation case in same execution → critical block

---

## 7. Observability

Every request produces a `Tracer` with:
- `trace_id`: `trc-<12 hex chars>` — unique per request
- `tool_calls[]`: ordered record of every component invoked
- `guardrail_events[]`: internal audit log of rule triggers

Tool call records include: `tool`, `action`, `status`, `latency_ms`, `meta`.

Structured JSON logging (configurable via `LOG_FORMAT=json|text`) emits every log line as a parseable JSON object for Datadog/Splunk/CloudWatch ingestion. Sensitive fields (API keys, tokens) are automatically redacted.

---

## 8. Security

- **Ownership enforcement**: `order.customer_id == session_customer_id` checked before every tool mutation. Implemented in `Executor._assert_ownership()`.
- **Error opacity**: `OwnershipError` returns `"Order not found"` — never reveals the real owner.
- **Session ID validation**: Pydantic regex `^[a-zA-Z0-9_\-]+$` rejects injection characters at the HTTP boundary.
- **No authentication (by spec)**: Session store maps `session_id → customer_id` via file + env var + embedded pattern.

---

## 9. Data Layer

All data is stored in JSON files under `data/`. Repositories maintain in-memory indexes (O(1) lookup) and flush to disk atomically using write-to-temp + `os.replace()` to prevent corruption.

File ownership:
- `orders.json` → OrderRepository (read/write)
- `crm_cases.json` → CrmRepository (read/write)
- `kb_articles.json` → KbRepository (read-only)
- `payment_config.json` → PaymentRepository (read-only)
- `refunds.json` → PaymentRepository (append-only)

---

## 10. Known Limitations

- JSON-backed storage is not suitable for concurrent multi-process deployments. Production would use PostgreSQL or DynamoDB.
- Session store is a simulation. Production would integrate with OAuth/JWT.
- LLM calls are synchronous within the async pipeline. Production would add circuit breakers and timeout budgets.