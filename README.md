# AtlasCare

Agentic AI customer support platform for Acme Retail Co. Customers can check orders, cancel items, request refunds, and get policy answers through a conversational interface. Built on FastAPI + LangGraph + Groq (Llama models) + Gradio.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- A [Groq API key](https://console.groq.com/keys) (free tier is sufficient)

---

## Installation

```bash
# 1. Clone the repo and enter the directory
git clone <repo-url>
cd AtlasCare

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy the example env file and fill in your Groq key:

```bash
cp .env.example .env
```

Open `.env` and set the required values:

```env
# Required — get your key at https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Models (these defaults work with the free Groq tier)
PLANNER_MODEL=llama-3.3-70b-versatile
RESPONSE_MODEL=llama-3.1-8b-instant
```

All other values in `.env.example` can be left as-is for local development.

---

## Running the Application

The application has three components. Start them in separate terminals.

### 1. API Server (required)

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Runs at `http://localhost:8000`. The interactive API docs are at `http://localhost:8000/docs`.

### 2. Customer Chat UI

```bash
python gradio_app.py
```

Runs at `http://localhost:7860`. This is the customer-facing chat interface — log in with one of the test accounts below, then chat with the agent.

### 3. Admin Dashboard (optional)

```bash
python admin_dashboard.py
```

Runs at `http://localhost:7861`. Shows live KPIs, trace logs, and tool call history. Requires the API server to be running.

---

## Test Accounts

All test users share the password `password`.

| Username | Customer ID | Notes                         |
|----------|-------------|-------------------------------|
| priya    | CUST-001    | Multiple orders in all states |
| arjun    | CUST-002    | Includes COD orders           |
| divya    | CUST-003    | High-value orders             |
| rahul    | CUST-004    | Escalation scenarios          |

---

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -q

# Run just the pre-launch suite (used as a startup gate)
python run_tests.py

# Skip slower edge-case tests
python run_tests.py --fast

# Run a single suite
python run_tests.py --suite guardrails
# Available suites: contracts, guardrails, security, journeys, edge_cases, regression
```

---

## Project Structure

```
AtlasCare/
├── main.py                  # FastAPI app — /query, /auth, /admin endpoints
├── gradio_app.py            # Customer chat UI
├── admin_dashboard.py       # Admin KPI and trace dashboard
├── agent/
│   ├── graph.py             # LangGraph agent — nodes, routing, state
│   └── guardrails.py        # Pre/post guardrail checks
├── tools/                   # OMS, CRM, Payment, KB tool wrappers
├── services/                # Auth, order, refund, escalation logic
├── repositories/            # JSON file persistence layer
├── models/                  # Pydantic request/response models
├── data/                    # JSON data files (orders, cases, KB articles, users)
├── observability/           # Structured logging and trace store
├── tests/                   # pytest test suites
└── requirements.txt
```

---

## Design Highlights

- **Dual-model routing**: simple queries use Llama 3.1 8B for speed (<3 s); complex or escalation-worthy queries route to Llama 3.3 70B for accuracy.
- **Deterministic guardrails first**: policy rules (refund threshold, message length, ownership) are enforced in Python before any LLM call.
- **History-aware ambiguous-query check**: vague messages like "what's the status of my order?" are intercepted and the customer is asked for their order ID — but only on the first mention. If a valid order ID already appears anywhere in the conversation history, the check is skipped and the LLM handles the follow-up normally.
- **Confirmation flow**: high-value cancellations (>₹5,000) pause the pipeline and ask the customer to confirm before the action is executed.
- **Evaluator node**: after response generation, a second LLM pass checks the response against the tool results and triggers a retry if data is missing or incorrect.

---

## Key Environment Variables

| Variable               | Description                                      | Default        |
|------------------------|--------------------------------------------------|----------------|
| `GROQ_API_KEY`         | Groq API key (required)                          | —              |
| `GROQ_BASE_URL`        | Groq API base URL                                | —              |
| `PLANNER_MODEL`        | 70B model for complex planning and evaluation    | llama-3.3-70b-versatile |
| `RESPONSE_MODEL`       | 8B model for fast response generation            | llama-3.1-8b-instant |
| `PORT`                 | API server port                                  | `8000`         |
| `GRADIO_PORT`          | Customer UI port                                 | `7860`         |
| `AUTO_REFUND_LIMIT_INR`| Max refund amount for autonomous processing      | `25000.0`      |
| `LOG_LEVEL`            | Logging level (`INFO`, `DEBUG`, `WARNING`)       | `INFO`         |
