# AtlasCare

Agentic AI customer support platform for Acme Retail Co. — check orders, cancel items, request refunds, and get policy answers through a chat interface. Built on FastAPI + LangGraph + Groq (Llama) + Gradio.

## Prerequisites

- Python 3.10–3.12
- A [Groq API key](https://console.groq.com/keys) (free tier works)

## Setup

```bash
git clone <repo-url> && cd AtlasCare

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

pip install -r requirements.txt

cp .env.example .env           # then set GROQ_API_KEY in .env
```

Only `GROQ_API_KEY` is required; every other value in `.env.example` works as-is for local use.

## Data — bring your own

You supply **four** canonical files in `data/`. They are the only inputs you provide:

| File | What it holds |
|------|---------------|
| `orders.json` | Orders, line items, statuses, payment methods |
| `crm_cases.json` | Customers (profiles, tiers, addresses) and existing cases |
| `kb_articles.json` | Policy articles, each tagged and scoped to product categories via `applies_to` |
| `payment_config.json` | Auto-refund limit, supported methods, refund SLA |

Each file must follow its schema in **`example_schema/`** (`schema_*.json`). To use your own data, drop your four files into `data/` and keep to those schemas — nothing else to wire up.

**Everything else is derived for you on startup**, as a pure function of those four files:

- `users.json`, `sessions.json` — one login + session per customer
- `refunds.json`, `order_audit_log.json` — empty runtime ledgers (never overwritten)
- `category_policies.json` — each product category → the policy articles that apply to it (inverts `applies_to`)
- `product_categories.json` — every product classified into a category by a deterministic keyword match

The **category list itself comes from `kb_articles.applies_to`** — it is not hardcoded. Add a new category (e.g. `garden`) to an article's `applies_to`, add a product that matches it, and it flows through automatically. Anything the classifier can't place lands in `misc`. To (re)generate the derived files by hand:

```bash
python -m data.derive_support_files            # fill in anything missing
python -m data.derive_support_files --force    # rebuild users/sessions/categories
```

### Verify your data first

```bash
python -m pytest tests/test_canonical_schema.py -q
```

If it fails, it names the exact file and field that violate `example_schema/`. Fix the data before running.

## Run

Start each in its own terminal:

```bash
uvicorn main:app --port 8000 --reload   # API (required) — docs at /docs
python gradio_app.py                     # Customer chat UI — http://localhost:7860
python admin_dashboard.py                # Admin KPIs/traces (optional) — http://localhost:7861
```

Log into the chat UI with a test account (all use password `password`):

| Username | Customer |
|----------|----------|
| priya    | CUST-001 |
| arjun    | CUST-002 |
| divya    | CUST-003 |
| rahul    | CUST-004 |
| sneha    | CUST-005 |

## Architecture

A request runs through an explicit LangGraph state machine — not a free-running tool loop. Each node has one job, and the edges between them are plain Python:

```
input_redaction → confirmation_check → pre_guardrail → policy_grounding
   → tool_agent → tool_executor → post_guardrail → responder → evaluator → END
```

- **input_redaction** — masks card/CVV/email/phone before anything sees the message.
- **confirmation_check** — resolves a pending "are you sure?" from the previous turn.
- **pre_guardrail** — deterministic checks before the model: over-limit refunds, fraud/safety, malformed order IDs.
- **policy_grounding** — a general policy question is answered from the knowledge base (and the in-context order's product **category**), skipping the planner. Order actions fall through to the tools.
- **tool_agent / tool_executor** — pick and run tools; a big model handles complex/mutating requests, a small fast one handles simple lookups.
- **post_guardrail** — a last money-safety net after tools run.
- **responder / evaluator** — write the reply from verified results; an LLM judge allows one retry.

The guiding rule: **the model proposes, code decides.** Every rule that touches money or changes an order is enforced in deterministic Python, not the prompt. Full write-up in [`docs/Architecture.md`](docs/Architecture.md).

## Tests

```bash
python -m pytest -q          # full suite
python run_tests.py --fast   # pre-launch gate, skips slow edge cases
```
