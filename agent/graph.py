import json
import logging
import operator
import os
import re
import time
from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from openai import AsyncOpenAI, BadRequestError

from agent.guardrails import Guardrails
from tools.oms_tool import OmsTool
from tools.crm_tool import CrmTool
from tools.payment_tool import PaymentTool, RefundThresholdError
from tools.kb_tool import KbTool
from observability.tracer import Tracer
from repositories.audit_repository import AuditRepository

logger = logging.getLogger(__name__)

_guardrails = Guardrails()
_oms        = OmsTool()
_crm        = CrmTool()
_payment    = PaymentTool()
_kb         = KbTool()
_audit      = AuditRepository()


class AtlasCareState(TypedDict):
    messages:              Annotated[list, operator.add]
    customer_id:           str
    session_id:            str
    guardrail_blocked:     bool
    execution_summary:     list[dict[str, Any]]
    tool_call_count:       int
    final_response:        str
    task_complete:         bool
    pending_action:        dict | None   # persists via checkpointer across turns
    awaiting_confirmation: bool          # persists via checkpointer across turns
    eval_retry_count:      int
    eval_feedback:         str
    eval_approved:         bool


TOOLS = [
    {"type": "function", "function": {
        "name": "get_order",
        "description": "Fetch a single order by ID. Returns status, items, tracking, and payment method.",
        "parameters": {"type": "object", "required": ["order_id"], "properties": {
            "order_id": {"type": "string", "description": "Order ID in format ORD-XXXXX (5 digits)"}
        }},
    }},
    {"type": "function", "function": {
        "name": "list_orders",
        "description": "List all orders for the authenticated customer.",
        "parameters": {"type": "object", "required": [], "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "cancel_item",
        "description": (
            "Cancel a single line item. Only works for placed/processing orders. "
            "IMPORTANT: line_id is the integer 'line_id' field from the order's items array — "
            "it is NOT the item's position. Always call get_order first to look up the correct "
            "line_id for the item the customer named before calling this tool. "
            "Pass refund_method if the customer requests a specific refund destination other than the original payment method."
        ),
        "parameters": {"type": "object", "required": ["order_id", "line_id"], "properties": {
            "order_id": {"type": "string"},
            "line_id":  {
                "type": "integer",
                "description": (
                    "The 'line_id' field of the specific item to cancel, taken from the "
                    "order's items array returned by get_order. Call get_order first."
                ),
            },
            "refund_method": {
                "type": "string",
                "enum": ["HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"],
                "description": "Override refund destination. Omit to refund to the original payment method.",
            },
        }},
    }},
    {"type": "function", "function": {
        "name": "process_refund",
        "description": "Process a refund. Max Rs.25,000 for autonomous processing.",
        "parameters": {"type": "object", "required": ["order_id", "amount_inr", "method"], "properties": {
            "order_id":   {"type": "string"},
            "amount_inr": {"type": "number"},
            "method":     {"type": "string",
                           "enum": ["HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "update_address",
        "description": "Update shipping address using a saved address label.",
        "parameters": {"type": "object", "required": ["order_id", "address_label"], "properties": {
            "order_id":      {"type": "string"},
            "address_label": {"type": "string", "description": "e.g. 'home' or 'office'"},
        }},
    }},
    {"type": "function", "function": {
        "name": "update_address_raw",
        "description": (
            "Update shipping address using a full address provided by the customer. "
            "Use this after the customer supplies address details in response to being "
            "told their requested label (e.g. 'office') is not saved in their profile."
        ),
        "parameters": {"type": "object", "required": ["order_id", "line1", "city", "state", "pincode"], "properties": {
            "order_id": {"type": "string"},
            "line1":    {"type": "string", "description": "Street address line"},
            "city":     {"type": "string"},
            "state":    {"type": "string"},
            "pincode":  {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "create_crm_case",
        "description": (
            "Create a support case in the CRM. Use escalate instead for urgent matters "
            "(fraud, legal threats, manager requests, account security). "
            "Use priority='high' for damaged, defective, or counterfeit items."
        ),
        "parameters": {"type": "object", "required": ["order_id", "reason"], "properties": {
            "order_id":   {"type": "string"},
            "reason":     {"type": "string"},
            "amount_inr": {"type": ["number", "null"]},
            "priority":   {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Case priority. Use 'high' for damaged/defective/counterfeit items.",
            },
        }},
    }},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Escalate to specialist team with a high-priority CRM case.",
        "parameters": {"type": "object", "required": ["order_id", "reason"], "properties": {
            "order_id":   {"type": "string"},
            "reason":     {"type": "string"},
            "amount_inr": {"type": ["number", "null"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "list_cases",
        "description": "List all support cases for the authenticated customer.",
        "parameters": {"type": "object", "required": [], "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search knowledge base for policy articles by tags.",
        "parameters": {"type": "object", "required": ["tags"], "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        }},
    }},
    {"type": "function", "function": {
        "name": "request_confirmation",
        "description": (
            "Request explicit user confirmation before executing a sensitive action. "
            "Use ONLY when: (1) cancelling an item whose unit_price is STRICTLY ABOVE ₹5,000 "
            "(i.e. unit_price > 5000 — items priced ₹200, ₹500, ₹1,000, ₹2,000, ₹4,999 do NOT qualify), "
            "or (2) the customer's description ambiguously matches two or more items in the order. "
            "Do NOT use for address updates, standard low-value cancellations, or refunds. "
            "If in doubt about whether the threshold is met, check: is unit_price > 5000? "
            "If no, cancel immediately without calling this tool. "
            "action and action_params must exactly match the tool call you would make if confirmed."
        ),
        "parameters": {
            "type": "object",
            "required": ["action", "action_params", "confirmation_message"],
            "properties": {
                "action":               {"type": "string", "description": "Tool name to run if confirmed."},
                "action_params":        {"type": "object", "description": "Exact params for that tool."},
                "confirmation_message": {"type": "string", "description": "Polite question to show the customer."},
            },
        },
    }},
]

_AGENT_SYSTEM = (
    "You are AtlasCare, a support assistant for Acme Retail Co. Be warm, patient, and genuinely helpful.\n"
    "Your name is AtlasCare — never use placeholder brackets like [Agent's Name] or [Name].\n"
    "Treat every customer message as an opportunity to assist — never be dismissive, and never imply\n"
    "the customer is repeating themselves. If they ask again, help again with the same care.\n"
    "Use the available tools to fulfill the customer's request.\n"
    "Prefer calling all needed tools in a single response when possible.\n"
    "If you call tools, do NOT write a customer reply — the response is generated from tool results.\n"
    "If no tools are needed, write a direct, helpful reply to the customer.\n"
    "NEVER write a planning narrative ('I will now cancel...', 'I am going to...', 'Let me proceed...'). "
    "When a tool call is required, call it immediately — do not announce or explain what you are about to do.\n"
    "Always fetch actual order data via tools before acting or responding — "
    "never rely solely on details the customer provides about their order.\n"
    # Bug 5: always use get_order when a specific order ID is present
    "When the customer's message contains a specific order ID (format ORD-XXXXX), ALWAYS use "
    "get_order to fetch that specific order — never use list_orders. "
    "list_orders is only for browsing all orders when no specific order ID is given.\n"
    "When cancelling an item the customer refers to by name, you MUST call get_order first "
    "to find the correct line_id for that item. Never guess a line_id.\n"
    "For refunds, default method to 'original' unless the customer specifies otherwise.\n"
    "For address updates: first try update_address with the label the customer mentioned. "
    "If it fails because the label is not found, tell the customer which labels are available "
    "and ask them to either use a saved label or provide the full address (street, city, state, pincode). "
    "Once they supply the details, use update_address_raw to apply the address directly.\n"
    "'Return' and 'refund' mean the same thing — always look up the order and help the customer.\n"
    # Bug 1: COD rule strengthened to code-level enforcement
    "COD REFUND RULE (MANDATORY):\n"
    "For any order with payment_method='COD':\n"
    "  - NEVER call process_refund with method='original' — cash cannot be refunded electronically.\n"
    "  - NEVER accept 'cash', 'same method', 'same payment method', or 'original' as a refund method.\n"
    "  - If the customer requests a return/refund and has NOT provided a specific electronic method "
    "(UPI, HDFC Credit Card, ICICI Debit Card, SBI Net Banking), ask for one BEFORE calling any tool.\n"
    "  - Only proceed with cancel_item or process_refund once you have an explicit electronic method.\n"
    # Bug 2: no re-call instruction
    "LIST_ORDERS RE-CALL RULE: If list_orders results are already present in the conversation "
    "context (i.e. a tool result with an 'orders' array has already been returned), do NOT call "
    "list_orders again. Apply any filter criteria directly to the data you already have — the "
    "response model will handle filtering. Calling list_orders a second time wastes a tool call "
    "and produces no new data.\n"
    "When the customer asks to filter, search, or browse orders by ANY criteria (date, month, year, "
    "status, amount, product name, or any combination) — call list_orders once to fetch all orders. "
    "The response model will apply the customer's filter criteria to the results.\n"
    # Bug 4: status cross-check
    "STATUS CROSS-CHECK (MANDATORY): After fetching an order with get_order, compare the actual "
    "order status in the data against any status the customer stated in their message. "
    "If they conflict (e.g. customer says 'delivered' but status is 'shipped', or customer says "
    "'not delivered' but status is 'delivered'), do NOT proceed with the action the customer "
    "requested based on their stated status. Instead, call NO further tools and respond directly, "
    "noting the discrepancy and explaining the correct next step based on the actual status. "
    "Example: customer says 'my delivered order ORD-10003' but status is 'shipped' → "
    "do not escalate or initiate a return; inform the customer the order is still in transit.\n"
    # Bug 6: escalation rules — add damaged/defective items
    "ESCALATION RULES — immediately call the escalate tool (no other action) when the customer:\n"
    "  - Claims they did not place an order, or the order was placed without their knowledge or consent.\n"
    "  - Reports fraud, unauthorized account activity, or account compromise.\n"
    "  - Reports physical injury, safety hazard, or dangerous product.\n"
    "  - Threatens legal action, mentions consumer court, police, or regulatory bodies.\n"
    "  - Explicitly asks to speak to a manager or senior agent.\n"
    "  - Reports harassment or abusive behaviour.\n"
    "  - Reports a delivered product that is damaged, defective, or counterfeit and wants a resolution.\n"
    "Do NOT cancel, refund, or take any other automated action in these cases — escalate only.\n"
    "When creating a CRM case (not an escalation) for a damaged, defective, or counterfeit item, "
    "always pass priority='high' to create_crm_case.\n"
    "CONFIRMATION RULE: Never generate a plain-text response asking the customer "
    "for confirmation (e.g. 'Are you sure?', 'Can you confirm?'). "
    "Call request_confirmation ONLY when ALL of the following are true: "
    "(1) the order status is 'placed' or 'processing' (cancellation is actually possible), AND "
    "(2) the item being cancelled has unit_price > 5000 (strictly greater — ₹5,000 itself does NOT qualify), "
    "OR the customer's description ambiguously matches two or more items in the order. "
    "If the order status is 'shipped', 'delivered', or 'cancelled', do NOT call request_confirmation — "
    "call cancel_item directly. The tool enforces business rules and will return the appropriate error. "
    "In every other case — including all refunds, address updates, and any item priced ₹5,000 or below — "
    "execute the action immediately without asking. "
    "Example: 'Laundry Mesh Bag' at ₹200 → cancel immediately. "
    "'Dell Laptop' at ₹55,000 in a placed order → call request_confirmation first. "
    "'Dell Laptop' at ₹55,000 in a shipped order → call cancel_item directly (will fail with error).\n"
)

_RESPONSE_SYSTEM = (
    "You are AtlasCare, a warm and helpful customer support assistant for Acme Retail Co.\n"
    "Write a clear, empathetic response using ONLY the data provided to you.\n"
    "Never invent order details, amounts, or tracking numbers.\n"
    "Never mention internal tool names, trace IDs, customer IDs, or any internal framing "
    "such as 'tool results', 'provided data', 'based on the data', or 'according to the system'.\n"
    "Speak naturally and directly as a customer support agent — never reference where the data came from.\n"
    "If the customer's message includes an explicit greeting word (hi, hello, hey, howdy, etc.), open your reply with a warm greeting in return. "
    "Otherwise, do NOT open with a greeting — go straight to the response. "
    "Short replies like 'Yes', 'No', 'ok', order IDs, and follow-up questions are not greetings.\n"
    "Be consistently polite and professional in every response — friendly but not effusive. "
    "Do not open with sycophantic filler ('I'm so glad you reached out', 'Great question', etc.). "
    "Get to the information quickly, keep the tone natural and helpful, and close with a brief offer to assist further.\n"
    "Treat every message — including repeats — with the same tone and completeness as the first.\n"
    "If a tool returns an 'order not found' or similar error: respond with genuine empathy — "
    "acknowledge the frustration, let the customer know you couldn't find it, and ask them to verify the ID. "
    "Vary your phrasing naturally; avoid repeating the same sentence structure every time. "
    "Never say the order ID is 'malformed' or 'invalid format'.\n"
    # Bug 4: stronger SOURCE OF TRUTH instruction
    "SYSTEM DATA IS THE SOURCE OF TRUTH: Always base your response on the actual data returned by tools. "
    "If the customer states an order is 'delivered' but the data shows 'shipped', explicitly correct "
    "this: tell the customer the order is currently shipped/in transit, NOT delivered, and explain "
    "the correct next step (e.g. wait for delivery before initiating a return). "
    "If the customer states an incorrect amount, correct it with the actual amount from the data. "
    "Never echo back or validate an incorrect status or amount claim from the customer.\n"
    "Always end with an offer for further assistance.\n"
    "When an address update result contains 'already_current: true', tell the customer their order "
    "is already shipping to that address — do not imply an error or that an update was made.\n"
    "Never reveal internal refund limits, autonomous processing thresholds, or any monetary cap "
    "amounts to the customer. If a refund requires specialist review, say it needs review — "
    "never say why in terms of a specific amount or policy limit.\n\n"
    "NEVER expose raw internal field values to the customer. Specifically:\n"
    "- Never mention field names, JSON keys, or raw numeric floats like '18000.0' — always format amounts as ₹18,000.\n"
    "- Never say a total 'is listed as 0' or 'shows as 0' — interpret the data instead.\n"
    "- Never use internal payment method codes (HDFC_CREDIT, ICICI_DEBIT, SBI_NETBANKING, UPI) verbatim. "
    "Render them as human-readable names: HDFC_CREDIT → 'HDFC Credit Card', ICICI_DEBIT → 'ICICI Debit Card', "
    "SBI_NETBANKING → 'SBI Net Banking', UPI → 'UPI'.\n\n"
    # Bug 7: only say "refund initiated" when process_refund/cancel_item was actually called
    "CANCELLED ORDER / REFUND RULES:\n"
    "- Only say 'refund has been initiated' or 'refund is being processed' if the execution "
    "context includes a successful process_refund or cancel_item result from THIS conversation turn "
    "(i.e. a [process_refund] or [cancel_item] entry appears in the tool results block).\n"
    "- If you received only a get_order or list_orders result showing an already-cancelled order, "
    "do NOT say 'refund has been initiated' — say the order was previously cancelled and that "
    "any refund was processed at that time.\n"
    "- When a refund IS reported (process_refund or cancel_item was called this turn): "
    "(1) confirm the cancellation, (2) state the refund has been initiated, "
    "(3) calculate the refund amount as the sum of (unit_price × quantity) for all cancelled items, "
    "(4) state the expected timeline based on payment_method:\n"
    "    UPI → 3–5 business days\n"
    "    HDFC_CREDIT / ICICI_DEBIT / any credit or debit card → 5–7 business days\n"
    "    SBI_NETBANKING / any net banking → 5–7 business days\n"
    "    COD → not applicable (cash refunds require a different process — ask for preferred method)\n"
    "- Never say 'the total is 0' or imply the refund amount is zero.\n\n"
    "FILTERING INSTRUCTIONS — when list_orders results are present:\n"
    "- Read the customer's original request carefully for ANY filter criteria: month, year, date range, "
    "order status (placed/processing/shipped/delivered/cancelled), amount range, product name, "
    "payment method, or any combination.\n"
    "- Apply ALL stated criteria to the order list before composing your reply. "
    "Only include orders that match every criterion the customer specified.\n"
    "- If no orders match the criteria, say so clearly and tell the customer what you searched.\n"
    "- If the customer gave no filter criteria, present all orders concisely.\n"
    "- Use the 'created_at' field for date/month/year filtering. "
    "The field is an ISO-8601 timestamp (e.g. '2026-06-15T10:30:00+00:00'). "
    "Parse month and year from it when needed.\n"
    "- When filtering by amount, compare against the 'total_amount' field."
)


_groq_client: AsyncOpenAI | None = None
_groq_client_key: str | None = None

def _get_groq_client() -> AsyncOpenAI:
    global _groq_client, _groq_client_key
    current_key = os.environ["GROQ_API_KEY"]
    if _groq_client is None or _groq_client_key != current_key:
        _groq_client     = AsyncOpenAI(api_key=current_key, base_url=os.environ["GROQ_BASE_URL"])
        _groq_client_key = current_key
    return _groq_client

class OwnershipError(Exception):
    pass

def _assert_ownership(order_customer_id: str, session_customer_id: str, order_id: str) -> None:
    if order_customer_id != session_customer_id:
        raise OwnershipError(f"Order {order_id} not found for the current session.")

async def _fetch_owned_order(oid: str, customer_id: str) -> dict:
    """Fetch an order and assert the session customer owns it."""
    order = await _oms.get_order(oid)
    _assert_ownership(order["customer_id"], customer_id, oid)
    return order

_METHOD_MAP = {
    "hdfc_credit": "HDFC_CREDIT", "icici_debit": "ICICI_DEBIT",
    "sbi_netbanking": "SBI_NETBANKING", "upi": "UPI", "original": "original",
    "hdfc": "HDFC_CREDIT", "hdfc credit": "HDFC_CREDIT", "hdfc credit card": "HDFC_CREDIT",
    "hdfc card": "HDFC_CREDIT", "icici": "ICICI_DEBIT", "icici debit": "ICICI_DEBIT",
    "icici debit card": "ICICI_DEBIT", "icici card": "ICICI_DEBIT",
    "sbi": "SBI_NETBANKING", "sbi net banking": "SBI_NETBANKING",
    "net banking": "SBI_NETBANKING", "netbanking": "SBI_NETBANKING",
    "gpay": "UPI", "google pay": "UPI", "phonepe": "UPI", "paytm": "UPI",
    "original_payment": "original", "original_payment_method": "original",
    "original payment method": "original", "same card": "original",
    "same method": "original", "source": "original",
}

def _normalise_refund_method(method: str) -> str:
    return _METHOD_MAP.get(method.lower().strip(), "original")

_VALID_ORDER_RE   = re.compile(r'\bORD-\d{5}\b', re.IGNORECASE)

def _extract_order_ids(text: str) -> list[str]:
    """Return all valid ORD-XXXXX IDs found by scanning for 'ORD-' and taking exactly 5 chars."""
    results, upper = [], text.upper()
    idx = 0
    while True:
        pos = upper.find("ORD-", idx)
        if pos == -1:
            break
        digits = upper[pos + 4: pos + 9]
        if len(digits) == 5 and digits.isdigit():
            results.append(f"ORD-{digits}")
        idx = pos + 4
    return results

def _clean_order_id(raw: str) -> str:
    """Extract the first valid ORD-XXXXX from raw LLM output (may contain trailing punctuation)."""
    ids = _extract_order_ids(raw)
    return ids[0] if ids else raw.strip().upper()

_DIRTY_ORDER_RE = re.compile(r'\bORD-\d{5}[^\w\s]*', re.IGNORECASE)

def _normalise_order_ids_in_text(text: str) -> str:
    """Replace every ORD-XXXXX<noise> occurrence with a clean ORD-XXXXX in a string."""
    return _DIRTY_ORDER_RE.sub(lambda m: m.group(0)[:9].upper(), text)

_INVALID_ORDER_RE = re.compile(
    r'\b(ORD-\d{1,4}|ORD-\d{6,}|ORD-[A-Za-z]+|ORDER-\w+)\b', re.IGNORECASE
)

def _check_order_id_format(message: str) -> str | None:
    if _VALID_ORDER_RE.search(message):
        return None
    m = _INVALID_ORDER_RE.search(message)
    if m:
        bad = m.group(0).upper()
        return (
            f"The order ID **{bad}** doesn't look right. "
            f"Order IDs follow the format **ORD-XXXXX** (5 digits), e.g. **ORD-78321**. "
            f"Could you please check and try again?"
        )
    return None


_LIST_INTENT_RE = re.compile(
    r'\b(list|show|display|see|view|get|find|fetch|what are|tell me|give me)\b.{0,30}\borders\b'
    r'|\borders\b.{0,20}\b(list|history|summary|all|recent|past|previous)\b'
    r'|\border\s+(history|list|summary)\b',
    re.IGNORECASE,
)

_AMBIGUOUS_ORDER_RE = re.compile(
    r'\b(my order|the order|an order|one of my orders?|this order|that order'
    r'|order details?|order (?:status|info(?:rmation)?|update|tracking)'
    r'|(?:info|information|details?|update|status|tracking)\s+(?:about|on|for)\s+(?:my|the|an)\s+order'
    r'|about\s+(?:my|an?|the)\s+order)\b',
    re.IGNORECASE,
)


def _is_ambiguous_order_query(message: str) -> bool:
    """Return True when the user is asking about a specific order but gave no order ID."""
    if _VALID_ORDER_RE.search(message):
        return False
    if _LIST_INTENT_RE.search(message):
        return False
    return bool(_AMBIGUOUS_ORDER_RE.search(message))


_COMPLEXITY_SIGNALS = frozenset([
    # Product condition
    "damaged", "defective", "broken", "not working", "faulty", "malfunctioning",
    "counterfeit", "fake", "duplicate", "not original", "tampered", "used product",
    "missing parts", "incomplete",

    # Delivery problems
    "never arrived", "not delivered", "not received", "never received",
    "missing package", "lost in transit", "wrong address delivered",
    "delivered to wrong", "partial delivery", "missing items",

    # Wrong order
    "wrong item", "wrong product", "wrong order", "not what i ordered",
    "different product",

    # Payment & financial disputes
    "double charged", "charged twice", "overcharged", "extra charge",
    "wrong amount", "money deducted", "payment failed but", "amount debited",
    "duplicate payment", "not refunded", "refund not received",

    # Safety & physical harm
    "injured", "hurt", "burn", "fire", "explosion", "electric shock",
    "dangerous", "hazardous", "unsafe",

    # Fraud & account security
    "fraud", "stolen", "never placed", "didn't place", "did not place",
    "not my order", "without my knowledge", "without my consent",
    "unauthorized", "not authorized", "someone else", "account hacked",
    "account compromised", "identity theft", "not me",

    # Escalation requests
    "manager", "supervisor", "escalate", "higher authority",
    "senior agent", "speak to someone",

    # Legal & regulatory
    "complaint", "legal", "lawsuit", "consumer court", "consumer forum",
    "police", "fir", "report to authorities", "consumer protection",
    "ombudsman",

    # Emotional distress
    "extremely frustrated", "very upset", "very angry", "disgusted",
    "harassment", "threatening", "abusive",
])

_MULTI_ACTION_VERBS = ["cancel", "refund", "return", "update", "change", "reship"]

# Bug 2: filter queries need 70B to reason over list_orders results without looping
_FILTER_SIGNALS = frozenset([
    "delivered orders", "shipped orders", "placed orders", "cancelled orders",
    "processing orders", "pending orders",
    "cod orders", "upi orders", "hdfc orders", "icici orders",
    "orders below", "orders above", "orders under", "orders over",
    "orders less than", "orders more than", "orders worth",
    "below rs", "above rs", "under rs", "over rs",
    "in january", "in february", "in march", "in april", "in may", "in june",
    "in july", "in august", "in september", "in october", "in november", "in december",
    "from last month", "this month", "last month", "from january", "from february",
    "from march", "from april", "from may", "from june", "from july", "from august",
    "from september", "from october", "from november", "from december",
    "orders placed in", "orders from", "filter", "show only", "sort by",
])

def _is_complex(message: str) -> bool:
    """Route to PLANNER_MODEL (70B) for complex, mutating, filter, or order-specific queries;
    fall back to RESPONSE_MODEL (8B) for plain lookups to hit <3 s end-to-end."""
    lower = message.lower()
    if any(sig in lower for sig in _COMPLEXITY_SIGNALS):
        return True
    # Bug 2: filter/aggregation over order list requires 70B to avoid re-call loop
    if any(sig in lower for sig in _FILTER_SIGNALS):
        return True
    action_count = sum(1 for v in _MULTI_ACTION_VERBS if v in lower)
    if action_count >= 2:
        return True
    if action_count >= 1 and _VALID_ORDER_RE.search(message):
        return True
    # Bug 5: any specific order ID → 70B picks get_order reliably
    if _VALID_ORDER_RE.search(message):
        return True
    return False


# Extend when adding tools whose results need filtering/ranking logic in the response.
_PLANNER_RESPONSE_TOOLS = {"list_orders", "list_cases", "get_order"}

_AFFIRMATIVE_RE = re.compile(
    r'^\s*(yes|yeah|yep|yup|ok|okay|sure|go\s+ahead|proceed|do\s+it|confirm|absolutely|fine|alright)\s*[.!]?\s*$',
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"^\s*(no|nope|nah|cancel|stop|don'?t|never\s*mind|skip|abort)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_READ_ONLY_TOOLS = frozenset({
    "get_order", "list_orders", "list_cases", "search_kb", "request_confirmation",
})

_MUTATING_TOOLS = frozenset({
    "cancel_item", "process_refund", "update_address", "update_address_raw",
    "create_crm_case", "escalate",
})

_CASUAL_RE = re.compile(
    r"\b(yo|hey|sup|hi|hello|howdy|hiya|what'?s up|gonna|wanna|gotta|lol|lmao|tbh|ngl|dunno|ya|yep|nope|cool|dude|bro)\b"
    r"|[!]{2,}|\b(where'?s|what'?s|how'?s|it'?s)\b",
    re.IGNORECASE,
)

def _tone_hint(message: str) -> str:
    """Return a one-line tone instruction based on the user's register."""
    if _CASUAL_RE.search(message):
        return "Match the customer's casual, relaxed tone — keep it friendly and informal."
    return "Use a warm, professional tone."


def _responder_model(user_msg: str, execution_summary: list[dict]) -> str:
    """Pick the model for response generation.

    Uses PLANNER_MODEL (70B) when the request was already classified as complex,
    or when the executed tools return multi-record data that needs reasoning.
    Falls back to RESPONSE_MODEL (8B) for simple single-record responses.
    """
    if _is_complex(user_msg):
        return os.environ["PLANNER_MODEL"]
    tools_called = {s["tool"] for s in execution_summary}
    if tools_called & _PLANNER_RESPONSE_TOOLS:
        return os.environ["PLANNER_MODEL"]
    return os.environ["RESPONSE_MODEL"]


async def _handle_get_order(args, customer_id, tracer):
    oid   = _clean_order_id(args["order_id"])
    order = await _fetch_owned_order(oid, customer_id)
    return {k: v for k, v in order.items() if k != "customer_id"}, "success", False

async def _handle_list_orders(args, customer_id, tracer):
    orders = await _oms.list_orders(customer_id)
    return {"orders": [{k: v for k, v in o.items() if k != "customer_id"} for o in orders]}, "success", False

async def _handle_cancel_item(args, customer_id, tracer):
    oid   = _clean_order_id(args["order_id"])
    order = await _fetch_owned_order(oid, customer_id)

    # Bug 3: short-circuit if all items are already cancelled
    all_items = order.get("items", [])
    if all_items and all(i.get("status") == "cancelled" for i in all_items):
        return {
            "info": (
                f"Order '{oid}' has already been fully cancelled. "
                "No further cancellation action is needed."
            )
        }, "success", False

    # Bug 1: COD gate — block if no explicit electronic refund method supplied
    original_method  = order.get("payment_method", "original")
    requested_method = args.get("refund_method")
    if original_method == "COD" and (not requested_method or requested_method == "original"):
        return {
            "error": (
                f"Order '{oid}' was paid via Cash on Delivery (COD). "
                "Cash cannot be refunded electronically. "
                "Please ask the customer to provide an electronic refund method: "
                "UPI (GPay / PhonePe / Paytm), HDFC Credit Card, "
                "ICICI Debit Card, or SBI Net Banking."
            )
        }, "error", False

    result = await _oms.cancel_item(oid, int(args["line_id"]))
    refund_amount    = result["unit_price"] * result["quantity"]
    payment_method   = requested_method if requested_method else original_method
    if payment_method == "COD":
        result["refund_note"] = "COD order — refund requires manual processing via a specialist."
    else:
        method = _normalise_refund_method(payment_method)
        try:
            result["refund"] = await _payment.process_refund(oid, refund_amount, method, customer_id)
        except Exception as exc:
            logger.warning("Auto-refund after cancel_item failed | order=%s | error=%s", oid, exc)
            result["refund_note"] = f"Item cancelled. Refund initiation failed: {exc}"

    audit_data: dict = {
        "line_id":         result["line_id"],
        "item_name":       result["name"],
        "unit_price":      result["unit_price"],
        "quantity":        result["quantity"],
        "new_order_total": result["new_order_total"],
    }
    if "refund" in result:
        audit_data["refund_id"]     = result["refund"].get("refund_id")
        audit_data["refund_amount"] = result["refund"].get("amount_inr")
        audit_data["refund_method"] = result["refund"].get("method")
    try:
        _audit.append(customer_id, oid, "item_cancelled", audit_data)
    except Exception as exc:
        logger.warning("Audit write failed | action=item_cancelled | order=%s | %s", oid, exc)

    return result, "success", False

async def _handle_process_refund(args, customer_id, tracer):
    oid    = _clean_order_id(args["order_id"])
    method = _normalise_refund_method(str(args.get("method", "original")))
    order  = await _fetch_owned_order(oid, customer_id)

    # Bug 3: block mutations on fully-cancelled zero-balance orders
    if order.get("status") == "cancelled":
        all_items = order.get("items", [])
        if all_items and all(i.get("status") == "cancelled" for i in all_items) \
                and float(order.get("total_amount", 0.0)) == 0.0:
            return {
                "error": (
                    f"Order '{oid}' has already been fully cancelled with no outstanding balance. "
                    "No further refund action is needed."
                )
            }, "error", False

    # Bug 1: COD gate — block if no explicit electronic method supplied
    if order.get("payment_method") == "COD" and method in {"original", "COD"}:
        return {
            "error": (
                f"Order '{oid}' was paid via Cash on Delivery (COD). "
                "A COD refund cannot be sent back as cash. "
                "Please ask the customer to specify an electronic refund method: "
                "UPI (GPay / PhonePe / Paytm), HDFC Credit Card, "
                "ICICI Debit Card, or SBI Net Banking."
            )
        }, "error", False

    # Block refunds on orders still being prepared — use cancel_item instead.
    if order.get("status") in {"placed", "processing"}:
        return {
            "error": (
                f"Order '{oid}' has status '{order['status']}' and cannot be refunded directly. "
                "Please cancel specific items via cancel_item first."
            )
        }, "error", False

    amount_inr  = float(args["amount_inr"])
    order_total = float(order.get("total_amount", 0.0))

    # Refund must not exceed what was actually charged for this order.
    if amount_inr > order_total:
        return {
            "error": (
                f"Requested refund of ₹{amount_inr:,.2f} exceeds the order total of "
                f"₹{order_total:,.2f} for order '{oid}'."
            )
        }, "error", False

    # Prevent duplicate refunds: sum of all previous refunds + this one must not exceed order total.
    already_refunded = _payment.get_total_refunded(oid)
    if already_refunded + amount_inr > order_total:
        remaining = order_total - already_refunded
        return {
            "error": (
                f"₹{already_refunded:,.2f} has already been refunded for order '{oid}'. "
                f"Maximum additional refund is ₹{remaining:,.2f}."
            )
        }, "error", False

    try:
        refund = await _payment.process_refund(oid, amount_inr, method, customer_id)
        try:
            _audit.append(customer_id, oid, "refund_processed", {
                "refund_id":  refund.get("refund_id"),
                "amount_inr": refund.get("amount_inr"),
                "method":     refund.get("method"),
                "escalated":  False,
            })
        except Exception as exc:
            logger.warning("Audit write failed | action=refund_processed | order=%s | %s", oid, exc)
        return {"refund": refund}, "success", False
    except RefundThresholdError:
        case = await _crm.create_case(
            customer_id=customer_id,
            order_id=oid,
            reason=f"High-value refund request of ₹{amount_inr:,.2f} requires specialist review.",
            amount_inr=amount_inr,
            trace_id=tracer.trace_id,
            priority="high",
        )
        try:
            _audit.append(customer_id, oid, "refund_processed", {
                "amount_inr": amount_inr,
                "method":     method,
                "escalated":  True,
                "case_id":    case["case_id"],
            })
        except Exception as exc:
            logger.warning("Audit write failed | action=refund_escalated | order=%s | %s", oid, exc)
        return {"case_id": case["case_id"], "escalated": True}, "success", True

async def _handle_update_address(args, customer_id, tracer):
    oid = _clean_order_id(args["order_id"])
    await _fetch_owned_order(oid, customer_id)
    result = await _oms.update_shipping_address(oid, customer_id, args["address_label"])
    if not result.get("already_current"):
        try:
            _audit.append(customer_id, oid, "address_updated", {
                "address_label": args["address_label"],
                "new_address":   result.get("new_address"),
            })
        except Exception as exc:
            logger.warning("Audit write failed | action=address_updated | order=%s | %s", oid, exc)
    return result, "success", False

async def _handle_update_address_raw(args, customer_id, tracer):
    oid = _clean_order_id(args["order_id"])
    await _fetch_owned_order(oid, customer_id)
    result = await _oms.update_shipping_address_raw(
        order_id=oid,
        line1=args["line1"],
        city=args["city"],
        state=args["state"],
        pincode=args["pincode"],
    )
    try:
        _audit.append(customer_id, oid, "address_updated", {
            "address_label": None,
            "new_address":   result.get("new_address"),
        })
    except Exception as exc:
        logger.warning("Audit write failed | action=address_updated_raw | order=%s | %s", oid, exc)
    return result, "success", False

async def _handle_create_crm_case(args, customer_id, tracer):
    oid      = _clean_order_id(args["order_id"])
    amt      = args.get("amount_inr")
    # Bug 6: honour priority passed by LLM (defaults to medium)
    priority = args.get("priority", "medium")
    case = await _crm.create_case(
        customer_id=customer_id, order_id=oid, reason=args["reason"],
        amount_inr=float(amt) if amt is not None else None,
        trace_id=tracer.trace_id, priority=priority,
    )
    return {"case": case}, "success", False

async def _handle_escalate(args, customer_id, tracer):
    oid = _clean_order_id(args["order_id"])
    amt = args.get("amount_inr")
    await _fetch_owned_order(oid, customer_id)
    case = await _crm.create_case(
        customer_id=customer_id, order_id=oid, reason=args["reason"],
        amount_inr=float(amt) if amt is not None else None,
        trace_id=tracer.trace_id, priority="high",
    )
    try:
        _audit.append(customer_id, oid, "escalation_created", {
            "case_id": case["case_id"],
            "reason":  args["reason"],
        })
    except Exception as exc:
        logger.warning("Audit write failed | action=escalation_created | order=%s | %s", oid, exc)
    return {"case_id": case["case_id"], "escalated": True}, "success", True

async def _handle_list_cases(args, customer_id, tracer):
    cases = await _crm.get_cases(customer_id)
    return {"cases": cases}, "success", False

async def _handle_search_kb(args, customer_id, tracer):
    articles = await _kb.search(tags=args.get("tags", []))
    return {"articles": articles}, "success", False

async def _handle_request_confirmation(args, customer_id, tracer):
    return {
        "confirmation_message": args.get("confirmation_message", "Can you confirm?"),
        "pending_action": {
            "tool":   args.get("action", ""),
            "params": args.get("action_params", {}),
        },
    }, "success", False


_TOOL_DISPATCH = {
    "get_order":            _handle_get_order,
    "list_orders":          _handle_list_orders,
    "cancel_item":          _handle_cancel_item,
    "process_refund":       _handle_process_refund,
    "update_address":       _handle_update_address,
    "update_address_raw":   _handle_update_address_raw,
    "create_crm_case":      _handle_create_crm_case,
    "escalate":             _handle_escalate,
    "list_cases":           _handle_list_cases,
    "search_kb":            _handle_search_kb,
    "request_confirmation": _handle_request_confirmation,
}


async def _dispatch_tool(
    tool_name: str,
    tool_input: dict,
    customer_id: str,
    tracer: Tracer,
) -> tuple[dict, str, bool]:
    """Returns (result_dict, status, escalated). Status: 'success'|'ownership_denied'|'error'."""
    try:
        handler = _TOOL_DISPATCH.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}, "error", False
        return await handler(tool_input, customer_id, tracer)
    except OwnershipError as exc:
        return {"error": str(exc)}, "ownership_denied", False
    except Exception as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return {"error": str(exc)}, "error", False


def _last_user_message(messages: list) -> str:
    return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")


async def confirmation_check_node(state: AtlasCareState, config) -> dict:
    """First node every turn. If a confirmation was pending, resolve it before anything else."""
    if not state.get("awaiting_confirmation"):
        return {}

    tracer      = config["configurable"]["tracer"]
    customer_id = state["customer_id"]
    user_msg    = _last_user_message(state["messages"])
    pending     = state.get("pending_action") or {}

    if _AFFIRMATIVE_RE.match(user_msg):
        t0 = time.monotonic()
        data, status, escalated = await _dispatch_tool(
            pending.get("tool", ""), pending.get("params", {}), customer_id, tracer,
        )
        tracer.record_tool_call(
            pending.get("tool", ""), pending.get("tool", ""), status,
            {"latency_ms": int((time.monotonic() - t0) * 1000)},
        )
        summary_entry = {
            "tool":         pending.get("tool", ""),
            "tool_call_id": "confirmation_dispatch",
            "success":      status == "success",
            "data":         data if status == "success" else {},
            "error":        data.get("error", "") if status != "success" else "",
            "escalated":    escalated,
        }
        return {
            "execution_summary":   state["execution_summary"] + [summary_entry],
            "tool_call_count":     state["tool_call_count"] + 1,
            "task_complete":       status == "success",
            "pending_action":      None,
            "awaiting_confirmation": False,
        }

    if _NEGATIVE_RE.match(user_msg):
        return {
            "pending_action":        None,
            "awaiting_confirmation": False,
            "final_response":        "Got it — I've cancelled that action. Is there anything else I can help you with?",
            "guardrail_blocked":     True,
        }

    return {"pending_action": None, "awaiting_confirmation": False}


def _route_confirmation_check(state: AtlasCareState) -> str:
    if state.get("guardrail_blocked"):
        return "end"
    if state.get("execution_summary"):   # affirmative dispatched an action
        return "post_guardrail"
    return "pre_guardrail"


async def pre_guardrail_node(state: AtlasCareState, config) -> dict:
    tracer: Tracer = config["configurable"]["tracer"]
    raw = _last_user_message(state["messages"])

    verdict = _guardrails.pre_check(raw, state["customer_id"], tracer)
    if verdict.blocked:
        return {"guardrail_blocked": True, "final_response": verdict.user_message, "task_complete": False}

    hint = _check_order_id_format(raw)
    if hint:
        return {"guardrail_blocked": True, "final_response": hint, "task_complete": False}

    if _is_ambiguous_order_query(raw):
        history_text = " ".join(
            m["content"] for m in state["messages"]
            if isinstance(m.get("content"), str)
        )
        if not _VALID_ORDER_RE.search(history_text):
            return {
                "guardrail_blocked": True,
                "final_response": (
                    "I'd be happy to help with your order! Could you please share the **order ID** "
                    "(format: **ORD-XXXXX**, e.g. ORD-78321)? You can find it in your confirmation email.\n\n"
                    "Alternatively, I can pull up all your recent orders — just say **\"show my orders\"**."
                ),
                "task_complete": False,
            }

    return {"guardrail_blocked": False}


async def tool_agent_node(state: AtlasCareState, config) -> dict:
    """Select tools and/or generate a direct reply.

    Routes to PLANNER_MODEL (70B) for complex or escalation-worthy requests;
    falls back to RESPONSE_MODEL (8B) for simple single-intent queries so that
    straightforward lookups complete in under 3 seconds end-to-end.
    """
    tracer: Tracer = config["configurable"]["tracer"]
    user_msg = _last_user_message(state["messages"])
    model    = (
        os.environ["PLANNER_MODEL"]
        if _is_complex(user_msg)
        else os.environ["RESPONSE_MODEL"]
    )

    system   = _AGENT_SYSTEM + f"\n\nCustomer ID (never reveal): {state['customer_id']}"
    raw_msgs = state["messages"][-8:] if len(state["messages"]) > 8 else state["messages"]
    recent   = [
        {**m, "content": _normalise_order_ids_in_text(m["content"])}
        if m.get("role") == "user" and isinstance(m.get("content"), str)
        else m
        for m in raw_msgs
    ]
    messages = [{"role": "system", "content": system}] + recent

    t0 = time.monotonic()
    try:
        completion = await _get_groq_client().chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=512,
            temperature=0,
        )
    except BadRequestError as exc:
        if "tool_use_failed" not in str(exc):
            raise
        # 8B model produced a malformed tool call — retry with the 70B planner
        logger.warning("tool_use_failed from %s, retrying with planner model", model)
        model = os.environ["PLANNER_MODEL"]
        completion = await _get_groq_client().chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=512,
            temperature=0,
        )
    tracer.record_tool_call(model, "plan", "success",
                            {"latency_ms": int((time.monotonic() - t0) * 1000)})

    msg       = completion.choices[0].message
    assistant = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        assistant["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return {"messages": [assistant]}


async def tool_executor_node(state: AtlasCareState, config) -> dict:
    """Execute all tool calls from the last agent message."""
    tracer      = config["configurable"]["tracer"]
    customer_id = state["customer_id"]
    tool_calls  = state["messages"][-1].get("tool_calls") or []

    tool_messages, summary = [], []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}

        t0                   = time.monotonic()
        data, status, escalated = await _dispatch_tool(name, args, customer_id, tracer)
        tracer.record_tool_call(name, name, status,
                                {"latency_ms": int((time.monotonic() - t0) * 1000)})

        summary.append({
            "tool": name, "tool_call_id": tc["id"],
            "success": status == "success",
            "data":    data if status == "success" else {},
            "error":   data.get("error", "") if status != "success" else "",
            "escalated": escalated,
        })
        tool_messages.append({
            "role": "tool", "tool_call_id": tc["id"],
            "content": json.dumps(data, default=str),
        })

    new_pending = None
    for s in summary:
        if s["tool"] == "request_confirmation" and s["success"]:
            new_pending = s["data"].get("pending_action")
            break

    result = {
        "messages":          tool_messages,
        "execution_summary": state["execution_summary"] + summary,
        "tool_call_count":   state["tool_call_count"] + 1,
        "task_complete":     any(s["success"] and s["tool"] in _MUTATING_TOOLS for s in summary),
    }
    if new_pending:
        result["pending_action"]        = new_pending
        result["awaiting_confirmation"] = True
    return result


async def post_guardrail_node(state: AtlasCareState, config) -> dict:
    tracer: Tracer = config["configurable"]["tracer"]
    verdict = _guardrails.post_check(state["execution_summary"], tracer)
    if verdict.blocked:
        return {"guardrail_blocked": True, "final_response": verdict.user_message}
    return {}


async def responder_node(state: AtlasCareState, config) -> dict:
    """Generate the customer-facing response."""
    tracer: Tracer = config["configurable"]["tracer"]

    # Escalation: deterministic response — no LLM call needed
    if any(s.get("escalated") for s in state["execution_summary"]):
        case_id = next(
            (s["data"].get("case_id", "pending")
             for s in state["execution_summary"] if s.get("escalated")),
            "pending",
        )
        return {
            "final_response": (
                "Thank you for reaching out. Your request requires specialist review.\n\n"
                f"I've created a priority support case (Case ID: **{case_id}**). "
                "A specialist will contact you within 24 hours.\n\n"
                "We apologise for any inconvenience."
            ),
        }

    # No tools were called this turn: agent answered directly from conversation history.
    # Reconstruct tool context from the most recent tool messages in history so the
    # responder can generate a properly-toned reply with verified data.
    if not state["execution_summary"]:
        prior_tool_lines = []
        for m in state["messages"]:
            if m.get("role") == "tool" and m.get("content"):
                prior_tool_lines.append(m["content"])
        if prior_tool_lines:
            tool_context = "\n".join(prior_tool_lines)
            user_req = _last_user_message(state["messages"])
            eval_feedback  = state.get("eval_feedback", "")
            eval_retry     = state.get("eval_retry_count", 0)
            feedback_prefix = (
                f"PREVIOUS RESPONSE REJECTED. Feedback: {eval_feedback}. "
                f"Generate an improved response that addresses this feedback.\n\n"
            ) if eval_feedback and eval_retry > 0 else ""
            resp_messages = [
                {"role": "system", "content": _RESPONSE_SYSTEM},
                {"role": "user", "content": (
                    f"{feedback_prefix}"
                    f"Tone: {_tone_hint(user_req)}\n"
                    f"Customer request: {user_req}\n\n"
                    f"Context:\n{tool_context}"
                )},
            ]
            t0 = time.monotonic()
            completion = await _get_groq_client().chat.completions.create(
                model=_responder_model(user_req, []),
                messages=resp_messages,
                max_tokens=1024,
                temperature=0.2,
            )
            tracer.record_tool_call("responder", "respond", "success",
                                    {"latency_ms": int((time.monotonic() - t0) * 1000)})
            text = (completion.choices[0].message.content or "").strip()
            return {"final_response": text or ""}
        # No prior tool results either — fall through to let the agent's direct text
        # be handled by the normal responder path below (execution_summary is empty
        # so tool_context will be blank, and the model generates a greeting/general reply).

    # Tools were called: build a clean, structured context for the response model
    # instead of passing the raw message history (tool_call IDs, JSON blobs, etc.)
    user_request = _last_user_message(state["messages"])
    # If the customer's last message was a plain affirmative (confirming a pending action),
    # recover the original request from earlier in the conversation so the response model
    # has meaningful context rather than just "Yes" to work from.
    if _AFFIRMATIVE_RE.match(user_request.strip()):
        for m in reversed(state["messages"][:-1]):
            if m.get("role") == "user" and not _AFFIRMATIVE_RE.match(m.get("content", "").strip()):
                user_request = m["content"]
                break
    tool_lines = []
    for s in state["execution_summary"]:
        if s["success"]:
            tool_lines.append(f"[{s['tool']}] {json.dumps(s['data'], default=str)}")
        else:
            tool_lines.append(f"[{s['tool']}] Error: {s['error']}")
    tool_context = "\n".join(tool_lines)

    eval_feedback   = state.get("eval_feedback", "")
    eval_retry      = state.get("eval_retry_count", 0)
    feedback_prefix = (
        f"PREVIOUS RESPONSE REJECTED. Feedback: {eval_feedback}. "
        f"Generate an improved response that addresses this feedback.\n\n"
    ) if eval_feedback and eval_retry > 0 else ""

    messages = [
        {"role": "system", "content": _RESPONSE_SYSTEM},
        {"role": "user", "content": (
            f"{feedback_prefix}"
            f"Tone: {_tone_hint(user_request)}\n"
            f"Customer request: {user_request}\n\n"
            f"Context:\n{tool_context}"
        )},
    ]

    t0         = time.monotonic()
    completion = await _get_groq_client().chat.completions.create(
        model=_responder_model(user_request, state["execution_summary"]),
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    tracer.record_tool_call("responder", "respond", "success",
                            {"latency_ms": int((time.monotonic() - t0) * 1000)})

    text = (completion.choices[0].message.content or "").strip()
    return {
        "final_response": text or "Your request has been processed. Is there anything else I can help with?",
    }


async def evaluator_node(state: AtlasCareState, config) -> dict:
    """Quality-check the responder's output. Bypasses for simple read-only queries."""
    tracer   = config["configurable"]["tracer"]
    user_msg = _last_user_message(state["messages"])
    # A plain affirmative ("yes", "ok", …) means the user confirmed a pending action.
    # The evaluator must judge the response against the *original* request, not "yes" —
    # otherwise it rejects a correct cancellation/refund response as unrelated to the
    # customer's message, triggering a retry that produces a garbled re-confirmation.
    if _AFFIRMATIVE_RE.match(user_msg.strip()):
        for m in reversed(state["messages"][:-1]):
            if m.get("role") == "user" and not _AFFIRMATIVE_RE.match(m.get("content", "").strip()):
                user_msg = m["content"]
                break

    # Bypass conditions — no LLM call
    if state.get("eval_retry_count", 0) >= 2:
        return {"eval_approved": True}
    if state.get("tool_call_count", 0) == 0 or not state.get("execution_summary"):
        # Only skip evaluation for genuinely conversational turns (no action needed).
        # If the user's message contains an action verb or an order ID, the agent
        # should have called tools — evaluate the response rather than auto-approving.
        if _is_complex(user_msg) or _VALID_ORDER_RE.search(user_msg):
            pass  # fall through to LLM evaluation below
        else:
            return {"eval_approved": True}
    if any(s.get("escalated") for s in state["execution_summary"]):
        return {"eval_approved": True}   # deterministic escalation response needs no check
    tools_called = {s["tool"] for s in state["execution_summary"]}
    if "request_confirmation" in tools_called:
        return {"eval_approved": True}   # confirmation prompt is self-verifying
    if tools_called.issubset(_READ_ONLY_TOOLS) and not _is_complex(user_msg):
        return {"eval_approved": True}

    tool_lines = [
        f"[{s['tool']}] {json.dumps(s['data'], default=str)}" if s["success"]
        else f"[{s['tool']}] Error: {s['error']}"
        for s in state["execution_summary"]
    ]
    eval_messages = [
        {"role": "system", "content": (
            "You are a quality checker for a customer support AI. Be strict.\n"
            "Evaluate whether the response correctly and completely addresses "
            "the customer's request using the tool results provided.\n"
            "Reply with exactly one of:\n"
            "  APPROVED\n"
            "  REJECTED: <one-sentence actionable feedback>\n"
            "Reject if: response omits important tool result data, invents details, "
            "uses internal field names, gives wrong amounts or statuses, or fails "
            "to address the customer's request.\n"
            "Reject if: the response uses future tense ('I can proceed', 'I will cancel', "
            "'the refund will be initiated', 'I'll update') for actions whose tool results "
            "show are already completed. Completed actions must be described in past tense.\n"
            "Reject if: the response describes a fabricated multi-step process not supported by "
            "the tool results (e.g. 'initiated to original method but updated to X' when only one "
            "refund call was made — the refund destination must match the actual method in the tool result).\n"
            # Bug 7: catch false "refund initiated" language
            "Reject if: the response says 'refund has been initiated', 'refund is being processed', "
            "or similar, but the tool results do NOT include a successful process_refund or "
            "cancel_item call. A get_order or list_orders result showing a pre-cancelled order "
            "does NOT mean a refund was initiated now."
        )},
        {"role": "user", "content": (
            f"Customer request: {user_msg}\n\n"
            f"Tool results:\n{chr(10).join(tool_lines)}\n\n"
            f"Response:\n{state.get('final_response', '')}"
        )},
    ]

    t0 = time.monotonic()
    completion = await _get_groq_client().chat.completions.create(
        model=os.environ["PLANNER_MODEL"],
        messages=eval_messages,
        max_tokens=128,
        temperature=0,
    )
    tracer.record_tool_call("evaluator", "evaluate", "success",
                            {"latency_ms": int((time.monotonic() - t0) * 1000)})

    verdict = (completion.choices[0].message.content or "").strip()
    if verdict.upper().startswith("APPROVED"):
        return {"eval_approved": True}

    feedback = verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
    return {
        "eval_approved":    False,
        "eval_feedback":    feedback,
        "eval_retry_count": state.get("eval_retry_count", 0) + 1,
    }


def _route_evaluator(state: AtlasCareState) -> str:
    if state.get("eval_approved") or state.get("eval_retry_count", 0) >= 2:
        return "end"
    return "responder"


def _route_tool_executor(state: AtlasCareState) -> str:
    """Loop back to tool_agent when only read-only lookups ran and the limit isn't reached.

    This lets the agent call get_order to discover a line_id, then immediately
    call cancel_item (or another mutating tool) in the same user turn — without
    needing the user to repeat themselves.

    Critical: once ANY mutating tool has been attempted (success OR failure),
    stop looping. Looping after a failed mutation lets the agent silently pivot
    to a different action the customer never requested.
    """
    if state["tool_call_count"] >= 3:
        return "post_guardrail"
    if state.get("awaiting_confirmation") or state.get("task_complete"):
        return "post_guardrail"
    # A mutating tool was already attempted — report the result, don't retry.
    if any(s["tool"] in _MUTATING_TOOLS for s in state.get("execution_summary", [])):
        return "post_guardrail"
    return "tool_agent"

def _route_pre_guardrail(state: AtlasCareState) -> str:
    return "end" if state["guardrail_blocked"] else "tool_agent"

def _route_tool_agent(state: AtlasCareState) -> str:
    last = state["messages"][-1]
    if last.get("tool_calls") and state["tool_call_count"] < 3:
        return "tools"
    return "respond"

def _route_post_guardrail(state: AtlasCareState) -> str:
    return "end" if state["guardrail_blocked"] else "responder"


def build_graph(checkpointer=None):
    g = StateGraph(AtlasCareState)

    g.add_node("confirmation_check", confirmation_check_node)
    g.add_node("pre_guardrail",      pre_guardrail_node)
    g.add_node("tool_agent",         tool_agent_node)
    g.add_node("tool_executor",      tool_executor_node)
    g.add_node("post_guardrail",     post_guardrail_node)
    g.add_node("responder",          responder_node)
    g.add_node("evaluator",          evaluator_node)

    g.add_edge(START, "confirmation_check")
    g.add_conditional_edges("confirmation_check", _route_confirmation_check,
                            {"end": END, "post_guardrail": "post_guardrail",
                             "pre_guardrail": "pre_guardrail"})
    g.add_conditional_edges("pre_guardrail",  _route_pre_guardrail,
                            {"end": END, "tool_agent": "tool_agent"})
    g.add_conditional_edges("tool_agent",     _route_tool_agent,
                            {"tools": "tool_executor", "respond": "post_guardrail"})
    g.add_conditional_edges("tool_executor", _route_tool_executor,
                            {"tool_agent": "tool_agent", "post_guardrail": "post_guardrail"})
    g.add_conditional_edges("post_guardrail", _route_post_guardrail,
                            {"end": END, "responder": "responder"})
    g.add_edge("responder", "evaluator")
    g.add_conditional_edges("evaluator",      _route_evaluator,
                            {"end": END, "responder": "responder"})

    return g.compile(checkpointer=checkpointer)
