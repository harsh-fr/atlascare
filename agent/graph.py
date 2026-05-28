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
from tools.payment_tool import PaymentTool
from tools.kb_tool import KbTool
from observability.tracer import Tracer

logger = logging.getLogger(__name__)

_guardrails = Guardrails()
_oms        = OmsTool()
_crm        = CrmTool()
_payment    = PaymentTool()
_kb         = KbTool()


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


#tools
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
            "line_id for the item the customer named before calling this tool."
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
        "name": "create_crm_case",
        "description": "Create a support case in the CRM.",
        "parameters": {"type": "object", "required": ["order_id", "reason"], "properties": {
            "order_id":   {"type": "string"},
            "reason":     {"type": "string"},
            "amount_inr": {"type": ["number", "null"]},
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
    "Always fetch actual order data via tools before acting or responding — "
    "never rely solely on details the customer provides about their order.\n"
    "When cancelling an item the customer refers to by name, you MUST call get_order first "
    "to find the correct line_id for that item. Never guess a line_id.\n"
    "For refunds, default method to 'original' unless the customer specifies otherwise.\n"
    "'Return' and 'refund' mean the same thing — always look up the order and help the customer.\n"
    "For COD (Cash on Delivery) orders: cash cannot be refunded electronically. "
    "Always ask the customer for a preferred electronic method (UPI, credit card, debit card, net banking) "
    "before calling process_refund on a COD order.\n"
    "When the customer asks to filter, search, or browse orders by ANY criteria (date, month, year, "
    "status, amount, product name, or any combination) — always call list_orders to fetch all orders. "
    "The response model will apply the customer's filter criteria to the results.\n"
    "ESCALATION RULES — immediately call the escalate tool (no other action) when the customer:\n"
    "  - Claims they did not place an order, or the order was placed without their knowledge or consent.\n"
    "  - Reports fraud, unauthorized account activity, or account compromise.\n"
    "  - Reports physical injury, safety hazard, or dangerous product.\n"
    "  - Threatens legal action, mentions consumer court, police, or regulatory bodies.\n"
    "  - Explicitly asks to speak to a manager or senior agent.\n"
    "  - Reports harassment or abusive behaviour.\n"
    "Do NOT cancel, refund, or take any other automated action in these cases — escalate only.\n"
    "CONFIRMATION RULE: Never generate a plain-text response asking the customer "
    "for confirmation (e.g. 'Are you sure?', 'Can you confirm?'). "
    "Call request_confirmation ONLY when the item being cancelled has unit_price > 5000 "
    "(strictly greater — ₹5,000 itself does NOT qualify), OR when the customer's description "
    "ambiguously matches two or more items in the order. "
    "In every other case — including all refunds, address updates, and any item priced ₹5,000 or below — "
    "execute the action immediately without asking. "
    "Example: 'Laundry Mesh Bag' at ₹200 → cancel immediately. "
    "'Dell Laptop' at ₹55,000 → call request_confirmation first.\n"
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
    "SYSTEM DATA IS THE SOURCE OF TRUTH: Always base your response on the actual data returned by tools. "
    "If the customer's description of any detail (status, delivery, amount, items, date, address, etc.) "
    "contradicts what the data shows, present the real data clearly and politely note the discrepancy — "
    "never echo back or validate unverified customer claims about order details.\n"
    "Always end with an offer for further assistance.\n\n"
    "NEVER expose raw internal field values to the customer. Specifically:\n"
    "- Never mention field names, JSON keys, or raw numeric floats like '18000.0' — always format amounts as ₹18,000.\n"
    "- Never say a total 'is listed as 0' or 'shows as 0' — interpret the data instead.\n"
    "- Never use internal payment method codes (HDFC_CREDIT, ICICI_DEBIT, SBI_NETBANKING, UPI) verbatim. "
    "Render them as human-readable names: HDFC_CREDIT → 'HDFC Credit Card', ICICI_DEBIT → 'ICICI Debit Card', "
    "SBI_NETBANKING → 'SBI Net Banking', UPI → 'UPI'.\n\n"
    "CANCELLED ORDER / REFUND RULES:\n"
    "- A cancelled order where every item is cancelled and total_amount is 0.0 means a refund has been initiated.\n"
    "- In this case: (1) confirm the cancellation, (2) state that the refund has been initiated, "
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

def _get_groq_client() -> AsyncOpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=os.environ["GROQ_BASE_URL"],
        )
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

def _normalise_order_ids_in_text(text: str) -> str:
    """Replace every ORD-XXXXX<noise> occurrence with a clean ORD-XXXXX in a string."""
    ids = _extract_order_ids(text)
    for oid in ids:
        # Replace the dirty form (ORD-XXXXX + any immediately attached non-word chars)
        text = re.sub(
            re.escape(oid) + r'[^\w\s]*',
            oid,
            text,
            flags=re.IGNORECASE,
        )
    return text

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


# Matches explicit list/browse intent — these should NOT trigger a clarification.
_LIST_INTENT_RE = re.compile(
    r'\b(list|show|display|see|view|get|find|fetch|what are|tell me|give me)\b.{0,30}\borders\b'
    r'|\borders\b.{0,20}\b(list|history|summary|all|recent|past|previous)\b'
    r'|\border\s+(history|list|summary)\b',
    re.IGNORECASE,
)

# Matches singular-order references that are non-specific (no order ID present).
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

def _is_complex(message: str) -> bool:
    """
    aim is to use PLANNER_MODEL (70B) for accurate tool selection.
    and           RESPONSE_MODEL (8B) for speed; target <3 s end-to-end.
    """
    lower = message.lower()
    if any(sig in lower for sig in _COMPLEXITY_SIGNALS):
        return True
    action_count = sum(1 for v in _MULTI_ACTION_VERBS if v in lower)
    if action_count >= 2:
        return True
    # Single mutating action on a specific order — needs 70B for accurate tool selection
    # (8B guesses line IDs without fetching the order first)
    if action_count >= 1 and _VALID_ORDER_RE.search(message):
        return True
    return False


# Tools whose results require non-trivial reasoning in the response (e.g. filtering,
# ranking, or cross-record analysis). Extend this set as new tools are added.
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


async def _dispatch_tool(
    tool_name: str,
    tool_input: dict,
    customer_id: str,
    tracer: Tracer,
) -> tuple[dict, str, bool]:
    """Returns (result_dict, status, escalated). Status: 'success'|'ownership_denied'|'error'."""
    try:
        if tool_name == "get_order":
            oid   = _clean_order_id(tool_input["order_id"])
            order = await _fetch_owned_order(oid, customer_id)
            return {k: v for k, v in order.items() if k != "customer_id"}, "success", False

        if tool_name == "list_orders":
            orders = await _oms.list_orders(customer_id)
            return {"orders": [{k: v for k, v in o.items() if k != "customer_id"} for o in orders]}, "success", False

        if tool_name == "cancel_item":
            oid   = _clean_order_id(tool_input["order_id"])
            order = await _fetch_owned_order(oid, customer_id)
            result = await _oms.cancel_item(oid, int(tool_input["line_id"]))
            # Automatically initiate refund for the cancelled item
            refund_amount = result["unit_price"] * result["quantity"]
            payment_method = order.get("payment_method", "original")
            if payment_method == "COD":
                result["refund_note"] = "COD order — refund requires manual processing via a specialist."
            else:
                method = _normalise_refund_method(payment_method)
                try:
                    refund = await _payment.process_refund(oid, refund_amount, method, customer_id)
                    result["refund"] = refund
                except Exception as exc:
                    logger.warning("Auto-refund after cancel_item failed | order=%s | error=%s", oid, exc)
                    result["refund_note"] = f"Item cancelled. Refund initiation failed: {exc}"
            return result, "success", False

        if tool_name == "process_refund":
            oid    = _clean_order_id(tool_input["order_id"])
            method = _normalise_refund_method(str(tool_input.get("method", "original")))
            order  = await _fetch_owned_order(oid, customer_id)
            # COD orders: "original" is invalid — cash cannot be refunded electronically.
            # The agent must ask the customer for an explicit electronic method.
            if order.get("payment_method") == "COD" and method == "original":
                return {
                    "error": (
                        "This order was paid via Cash on Delivery (COD). "
                        "A COD refund cannot be sent back as cash. "
                        "Please ask the customer to specify an electronic refund method: "
                        "UPI (GPay / PhonePe / Paytm), HDFC Credit Card, "
                        "ICICI Debit Card, or SBI Net Banking."
                    )
                }, "error", False
            refund = await _payment.process_refund(oid, float(tool_input["amount_inr"]), method, customer_id)
            return {"refund": refund}, "success", False

        if tool_name == "update_address":
            oid = _clean_order_id(tool_input["order_id"])
            await _fetch_owned_order(oid, customer_id)
            result = await _oms.update_shipping_address(oid, customer_id, tool_input["address_label"])
            return result, "success", False

        if tool_name == "create_crm_case":
            oid = _clean_order_id(tool_input["order_id"])
            amt = tool_input.get("amount_inr")
            case = await _crm.create_case(
                customer_id=customer_id, order_id=oid, reason=tool_input["reason"],
                amount_inr=float(amt) if amt is not None else None,
                trace_id=tracer.trace_id,
            )
            return {"case": case}, "success", False

        if tool_name == "escalate":
            oid = _clean_order_id(tool_input["order_id"])
            amt = tool_input.get("amount_inr")
            await _fetch_owned_order(oid, customer_id)
            case = await _crm.create_case(
                customer_id=customer_id, order_id=oid, reason=tool_input["reason"],
                amount_inr=float(amt) if amt is not None else None,
                trace_id=tracer.trace_id, priority="high",
            )
            return {"case_id": case["case_id"], "escalated": True}, "success", True

        if tool_name == "list_cases":
            cases = await _crm.get_cases(customer_id)
            return {"cases": cases}, "success", False

        if tool_name == "search_kb":
            articles = await _kb.search(tags=tool_input.get("tags", []))
            return {"articles": articles}, "success", False

        if tool_name == "request_confirmation":
            return {
                "confirmation_message": tool_input.get("confirmation_message", "Can you confirm?"),
                "pending_action": {
                    "tool":   tool_input.get("action", ""),
                    "params": tool_input.get("action_params", {}),
                },
            }, "success", False

        return {"error": f"Unknown tool: {tool_name}"}, "error", False

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

    # User changed topic — clear confirmation state and proceed normally
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
        # request_confirmation means action is still pending — not complete yet
        "task_complete":     any(s["success"] and s["tool"] != "request_confirmation" for s in summary),
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
            "task_complete": True,
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
            return {"final_response": text or "", "task_complete": False}
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
        "task_complete":  any(s["success"] for s in state["execution_summary"]),
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
        return {"eval_approved": True}
    if any(s.get("escalated") for s in state["execution_summary"]):
        return {"eval_approved": True}   # deterministic escalation response needs no check
    tools_called = {s["tool"] for s in state["execution_summary"]}
    if "request_confirmation" in tools_called:
        return {"eval_approved": True}   # confirmation prompt is self-verifying
    if tools_called.issubset(_READ_ONLY_TOOLS) and not _is_complex(user_msg):
        return {"eval_approved": True}   # J1 latency path preserved

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
            "to address the customer's request."
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


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_evaluator(state: AtlasCareState) -> str:
    if state.get("eval_approved") or state.get("eval_retry_count", 0) >= 2:
        return "end"
    return "responder"


def _route_pre_guardrail(state: AtlasCareState) -> str:
    return "end" if state["guardrail_blocked"] else "tool_agent"

def _route_tool_agent(state: AtlasCareState) -> str:
    last = state["messages"][-1]
    if last.get("tool_calls") and state["tool_call_count"] < 3:
        return "tools"
    return "respond"

def _route_post_guardrail(state: AtlasCareState) -> str:
    return "end" if state["guardrail_blocked"] else "responder"


# ── Graph builder ─────────────────────────────────────────────────────────────

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
    g.add_edge("tool_executor", "post_guardrail")
    g.add_conditional_edges("post_guardrail", _route_post_guardrail,
                            {"end": END, "responder": "responder"})
    g.add_edge("responder", "evaluator")
    g.add_conditional_edges("evaluator",      _route_evaluator,
                            {"end": END, "responder": "responder"})

    return g.compile(checkpointer=checkpointer)
