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
│               agent/graph.py  (LangGraph)           │
│  Compiled StateGraph — wires all pipeline nodes     │
│                                                     │
│  pre_guardrail → tool_agent → tool_executor         │
│       └─(blocked)→ END         │                   │
│                                ▼                   │
│                         post_guardrail              │
│                       └─(blocked)→ END             │
│                                │                   │
│                                ▼                   │
│                           responder → END           │
└──────────────────────┬──────────────────────────────┘
                       │  tool_executor calls
          ┌────────────┼────────────┬────────────┐
          ▼            ▼            ▼            ▼
     oms_tool.py  crm_tool.py  payment_tool.py  kb_tool.py
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
order_  crm_  payment_  kb_
repo    repo   repo     repo
    │
  data/*.json  (JSON-backed, atomic writes)
```

---

## 3. Request Pipeline

| Step | Node | Model | What happens |
|------|------|-------|--------------|
| 1 | main.py | — | `session_id` → `customer_id` via SessionStore |
| 2 | `pre_guardrail` | — (code) | GR-001/002/003 checks; order ID format validation |
| 3 | `tool_agent` | **Llama 3.3 70B** (complex) or **8B** (simple) | Selects tools to call, or writes a direct reply |
| 4 | `tool_executor` | — (code) | Dispatches each tool call with ownership validation |
| 5 | `post_guardrail` | — (code) | GR-004: verifies no payment on escalation case |
| 6 | `responder` | **Llama 3.1 8B Instant** | Formats tool results into a natural customer reply |

For no-tool requests (greetings, chitchat), `tool_agent` writes a direct reply and the pipeline skips `tool_executor` → `post_guardrail` → goes straight to `responder`, which uses the agent's text as-is (no second LLM call).

For escalations, `responder` uses a deterministic template (no LLM call).

---

## 4. LLM Model Routing

Two Llama models are used via the Groq API (OpenAI-compatible endpoint). Using both is intentional — planning accuracy and response latency have different requirements.

| Pipeline node | Model | Rationale |
|---|---|---|
| `tool_agent` — complex queries | **Llama 3.3 70B Versatile** | Multi-step tool selection; accuracy matters |
| `tool_agent` — simple queries | **Llama 3.1 8B Instant** | Single-intent lookups; speed matters |
| `responder` — tool-using paths | **Llama 3.1 8B Instant** | Formats verified data; reasoning depth not needed |
| `responder` — escalation | — (deterministic template) | Consistent, auditable output |
| `responder` — no-tool path | — (reuse tool_agent output) | Avoids a second LLM call entirely |

Complexity classifier (`_is_complex`): messages containing escalation signals (damaged, fraud, lawsuit, etc.) or two or more action verbs (cancel + refund) route to the 70B model.

**Config**: `PLANNER_MODEL` and `RESPONSE_MODEL` env vars — set in `.env`, pointing to `api.groq.com/openai/v1`.

---

## 5. Tool Design

Tools are typed async interfaces. They are the **only** layer that touches repositories. Agent code never accesses JSON files directly.

| Tool | Key methods | Backed by |
|------|------------|-----------|
| OmsTool | get_order, list_orders, cancel_item, update_shipping_address | OrderRepository |
| CrmTool | get_customer, create_case, get_cases | CrmRepository |
| PaymentTool | process_refund | PaymentRepository |
| KbTool | search, get_article | KbRepository |

Tools are swappable — replacing JSON repos with REST APIs requires only changing the repository layer.

---

## 6. Guardrails (Defence in Depth)

Three independent layers enforce the Rs.25,000 threshold:

| Layer | Where | When |
|-------|-------|------|
| GR-001 | `pre_guardrail` node | Before LLM — regex extracts amount from message |
| `PaymentTool._enforce_threshold` | payment_tool.py | At call time — Decimal comparison |
| GR-004 | `post_guardrail` node | After execution — verifies no payment on escalation case |

**Rule inventory:**
- GR-001: High-value refund mention → block + escalation message
- GR-002: Empty message → reject
- GR-003: Message > 2,000 chars → reject (prompt injection defence)
- GR-004: Payment success + escalation case in same turn → critical block

---

## 7. Observability

Every request produces a `Tracer` with:
- `trace_id`: `trc-<12 hex chars>` — unique per request
- `tool_calls[]`: ordered record of every component invoked (model calls, tool calls, guardrail triggers)

Structured JSON logging (`LOG_FORMAT=json|text`) emits every log line as a parseable JSON object.

The `TraceStore` is an in-memory ring buffer (last 500 traces) exposed via `/admin/traces` and `/admin/kpis`.

---

## 8. Security

- **Ownership enforcement**: `order.customer_id == session_customer_id` checked before every tool mutation. Implemented in `agent/graph.py:_assert_ownership()`.
- **Error opacity**: `OwnershipError` returns `"Order not found"` — never reveals the real owner.
- **Session ID validation**: Pydantic regex `^[a-zA-Z0-9_\-]+$` rejects injection characters at the HTTP boundary.
- **No authentication (by spec)**: Session store maps `session_id → customer_id` via `data/sessions.json` and an embedded-pattern extractor for auth-generated sessions.

---

## 9. Data Layer

All data is stored in JSON files under `data/`. Repositories maintain in-memory indexes (O(1) lookup) and flush to disk atomically using write-to-temp + `os.replace()` to prevent corruption.

| File | Repository | Access |
|------|------------|--------|
| `orders.json` | OrderRepository | read/write |
| `crm_cases.json` | CrmRepository | read/write |
| `kb_articles.json` | KbRepository | read-only |
| `payment_config.json` | PaymentRepository | read-only |
| `refunds.json` | PaymentRepository | append-only |
| `sessions.json` | SessionStore | read-only |
| `users.json` | UserRepository | read/write |

---

## 10. API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/query` | Submit a customer message to the agent |
| GET | `/health` | Liveness probe |
| POST | `/cases` | Create a CRM support case directly |
| GET | `/kb/search?tags=...` | Search KB articles by comma-separated tags |
| GET | `/admin/traces` | Fetch recent request traces |
| GET | `/admin/kpis` | Fetch KPI summary |
| DELETE | `/session/{session_id}` | Clear server-side session history |
| POST | `/auth/login` | Authenticate user, receive session_id |
| POST | `/auth/register` | Register a new user account |
| POST | `/auth/request-otp` | Request a password-reset OTP |
| POST | `/auth/reset-password` | Reset password with OTP |

---

## 11. Known Limitations

- JSON-backed storage is not suitable for concurrent multi-process deployments. Production would use PostgreSQL or DynamoDB.
- Session store is a simulation. Production would integrate with OAuth/JWT.
- `MemorySaver` checkpointer stores conversation history in-process memory only — restarting the server clears all histories.
- LLM calls are not circuit-broken. Production would add timeout budgets and fallback paths.
