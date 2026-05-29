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

> The app reads four data files from `data/` (`crm_cases`, `orders`, `kb_articles`, `payment_config`). On startup it auto-generates the remaining support files (`users`, `sessions`, refund/audit ledgers) from them — no manual data setup needed.

## Verify the data

Before starting the app, confirm the four data files conform to the expected schemas:

```bash
python -m pytest tests/test_canonical_schema.py -q
```

If this fails, it prints exactly which file and field violate `example_schema/`. Fix the data before running.

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

## Tests

```bash
python -m pytest -q          # full suite
python run_tests.py --fast   # pre-launch gate, skips slow edge cases
```
