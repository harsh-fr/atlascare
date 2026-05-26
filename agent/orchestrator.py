"""
agent/orchestrator.py
=====================
Central coordinator for AtlasCare.

Responsibility chain
--------------------
  1. Resolve session → customer identity
  2. Pre-execution guardrails (code-enforced)
  3. Vague help detection — ask for order ID gracefully
  4. Order ID format validation — catch malformed IDs pre-LLM
  5. Plan via LLM (Gemini)
  6. Auto-fill missing refund amounts from order data
  7. Execute plan (deterministic tool dispatch)
  8. Post-execution guardrails
  9. Build response (LLM-assisted or fast-path)

Conversation memory
-------------------
  Per-session history (last 10 turns) is maintained in a module-level
  dict keyed by session_id. The last 4 turns are prepended to the
  planner message so the LLM has context across turns.

  NOTE: The Gradio frontend ALSO sends context as a prefix. The
  orchestrator strips that prefix before storing it in its own history
  to avoid double-storing the context block.
"""

import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass

from agent.planner import Planner, ActionPlan
from agent.executor import Executor, ExecutionResult
from agent.guardrails import Guardrails, GuardrailVerdict
from agent.response_builder import ResponseBuilder
from observability.tracer import Tracer
from utils.session_store import SessionStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-session conversation history  (module-level, survives across requests)
# key  : session_id
# value: deque of {"role": "user"|"assistant", "content": str}
# ---------------------------------------------------------------------------
_session_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorResult:
    """Thin result envelope returned to the HTTP layer."""
    response_text: str
    task_complete: bool = False


def clear_session_history(session_id: str) -> None:
    """Remove all server-side conversation history for a terminated session."""
    _session_history.pop(session_id, None)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """
    Wires together the full agent pipeline for a single request.
    Stateless across requests — all per-request state lives in the
    Tracer and local variables.
    """

    def __init__(self) -> None:
        self._session_store    = SessionStore()
        self._guardrails       = Guardrails()
        self._planner          = Planner()
        self._executor         = Executor()
        self._response_builder = ResponseBuilder()
        logger.debug("Orchestrator initialised.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def handle(
        self,
        message: str,
        session_id: str,
        tracer: Tracer,
    ) -> OrchestratorResult:

        # ── Step 1: Session → customer ─────────────────────────────────
        customer_id = self._session_store.resolve(session_id)
        if customer_id is None:
            logger.warning("Unresolvable session | trace=%s | session=%s",
                           tracer.trace_id, session_id)
            return OrchestratorResult(
                response_text=(
                    "I'm unable to verify your session. "
                    "Please log in again and retry."
                )
            )
        tracer.set_customer_id(customer_id)

        # Strip any context prefix injected by the Gradio frontend
        # so we store only the raw user message in history
        raw_message = self._strip_context_prefix(message)

        # ── Step 2: Pre-guardrails ──────────────────────────────────────
        verdict: GuardrailVerdict = self._guardrails.pre_check(
            message=raw_message,
            customer_id=customer_id,
            tracer=tracer,
        )
        if verdict.blocked:
            logger.warning("Guardrail blocked | trace=%s | rule=%s",
                           tracer.trace_id, verdict.rule_id)
            return OrchestratorResult(response_text=verdict.user_message)

        # ── Step 3a: Greeting detection — route to LLM for natural reply ──
        if self._is_greeting(raw_message, session_id):
            history = _session_history[session_id]
            try:
                response = await self._response_builder.converse(
                    raw_message, list(history), tracer
                )
            except Exception:
                response = "Hello! Welcome to AtlasCare. How can I help you today?"
            self._store_turn(session_id, raw_message, response)
            return OrchestratorResult(response_text=response, task_complete=False)

        # ── Step 3b: Vague help detection (pre-LLM, deterministic) ────────
        if self._is_vague_help(raw_message, session_id):
            response = (
                "Of course, I'm happy to help! 😊\n\n"
                "Could you please share your **Order ID** so I can look "
                "into this for you? It's a number like **ORD-78321** "
                "(ORD- followed by 5 digits) and can be found in your "
                "order confirmation email or order history page."
            )
            self._store_turn(session_id, raw_message, response)
            return OrchestratorResult(response_text=response, task_complete=False)

        # ── Step 4: Order ID format check (pre-LLM, deterministic) ─────
        # Only fires when message contains something that looks like a
        # malformed order ID. Messages with no order ID pass through.
        format_hint = self._check_order_id_format(raw_message)
        if format_hint:
            self._store_turn(session_id, raw_message, format_hint)
            return OrchestratorResult(response_text=format_hint)

        # ── Step 5: Build enriched message with conversation history ────
        history       = _session_history[session_id]
        planner_input = self._build_planner_input(raw_message, history)

        # ── Step 6: Plan ────────────────────────────────────────────────
        try:
            plan: ActionPlan = await self._planner.plan(
                message=planner_input,
                customer_id=customer_id,
                tracer=tracer,
            )
        except Exception as exc:
            logger.exception("Planner failure | trace=%s | error=%s",
                              tracer.trace_id, exc)
            return OrchestratorResult(
                response_text=(
                    "I encountered an issue understanding your request. "
                    "Please rephrase and try again."
                )
            )

        logger.info("Plan produced | trace=%s | intent=%s | steps=%d",
                    tracer.trace_id, plan.intent, len(plan.steps))

        # ── Step 7: Auto-fill missing refund amounts ────────────────────
        plan = await self._auto_fill_refund_amount(plan, customer_id)

        # ── Step 8: Execute ─────────────────────────────────────────────
        try:
            execution_result: ExecutionResult = await self._executor.execute(
                plan=plan,
                customer_id=customer_id,
                tracer=tracer,
            )
        except Exception as exc:
            logger.exception("Executor failure | trace=%s | error=%s",
                              tracer.trace_id, exc)
            return OrchestratorResult(
                response_text=(
                    "Something went wrong while processing your request. "
                    "Our team has been notified. Please try again shortly."
                )
            )

        # ── Step 9: Post-guardrails ─────────────────────────────────────
        post_verdict: GuardrailVerdict = self._guardrails.post_check(
            execution_result=execution_result,
            tracer=tracer,
        )
        if post_verdict.blocked:
            logger.error("Post-guardrail triggered | trace=%s | rule=%s",
                         tracer.trace_id, post_verdict.rule_id)
            return OrchestratorResult(response_text=post_verdict.user_message)

        # ── Step 10: Build response ─────────────────────────────────────
        try:
            response_text = await self._response_builder.build(
                message=raw_message,
                plan=plan,
                execution_result=execution_result,
                customer_id=customer_id,
                tracer=tracer,
            )
        except Exception as exc:
            logger.exception("ResponseBuilder failure | trace=%s | error=%s",
                              tracer.trace_id, exc)
            response_text = execution_result.fallback_summary()

        # ── Store turn in session history ───────────────────────────────
        self._store_turn(session_id, raw_message, response_text)

        return OrchestratorResult(
            response_text=response_text,
            task_complete=execution_result.any_success(),
        )

    # ------------------------------------------------------------------
    # Conversation history helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _store_turn(session_id: str, user_msg: str, bot_msg: str) -> None:
        """Append a completed turn to the session history."""
        _session_history[session_id].append({"role": "user",      "content": user_msg})
        _session_history[session_id].append({"role": "assistant",  "content": bot_msg})

    @staticmethod
    def _build_planner_input(raw_message: str, history: deque) -> str:
        """
        Build the string sent to the planner LLM.

        Prepends the last 4 history turns (2 exchanges) as context so
        the planner can resolve pronouns like "this order" or
        "the payment method" back to a previously discussed order ID.

        Kept to 4 turns (not more) to avoid inflating the token budget.
        """
        if not history:
            return raw_message

        recent = list(history)[-4:]
        lines  = ["[Conversation context — use this to resolve references:]"]
        for turn in recent:
            role = "Customer" if turn["role"] == "user" else "Agent"
            # Truncate long agent responses to save tokens
            content = turn["content"][:200] if turn["role"] == "assistant" else turn["content"][:300]
            lines.append(f"{role}: {content}")
        lines.append(f"\n[Current customer message:] {raw_message}")
        return "\n".join(lines)

    @staticmethod
    def _strip_context_prefix(message: str) -> str:
        """
        Remove the context prefix injected by the Gradio frontend
        (lines starting with [Recent conversation for context:] ... [Current message:])
        so we store only the clean user message in server-side history.
        """
        marker = "[Current message:]"
        if marker in message:
            return message.split(marker, 1)[-1].strip()

        # Also handle the planner's own prefix format
        marker2 = "[Current customer message:]"
        if marker2 in message:
            return message.split(marker2, 1)[-1].strip()

        return message

    # ------------------------------------------------------------------
    # Greeting detection — route to LLM for natural conversation
    # ------------------------------------------------------------------
    _GREETING_PATTERNS = re.compile(
        r"^\s*("
        r"hi|hello|hey|good morning|good afternoon|good evening|"
        r"howdy|greetings|what'?s up|sup"
        r")\s*[.!?]?\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _is_greeting(cls, message: str, session_id: str) -> bool:
        """
        True when the message is a pure greeting with no order ID context.
        These are routed to the LLM for a natural conversational response.
        """
        stripped = message.strip()
        if re.search(r'\bORD[-\s]?\d', stripped, re.IGNORECASE):
            return False
        if not cls._GREETING_PATTERNS.match(stripped):
            return False
        # Once a conversation is in progress (order ID seen), don't intercept
        history = _session_history[session_id]
        if history:
            combined = " ".join(t["content"] for t in history)
            if re.search(r'\bORD-\d{5}\b', combined, re.IGNORECASE):
                return False
        return True

    # ------------------------------------------------------------------
    # Vague help detection
    # ------------------------------------------------------------------
    _VAGUE_PATTERNS = re.compile(
        r"^\s*("
        r"help|i need help|need help|i have an issue|i have a problem|"
        r"i need assistance|can you help|can you help me|"
        r"need help with my order|help with order|help with my order|"
        r"i need support|support"
        r")\s*[.!?]?\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _is_vague_help(cls, message: str, session_id: str) -> bool:
        """
        Returns True only when:
        1. Message matches a vague help pattern, AND
        2. There is NO prior context in session history that already
           contains an order ID (so we don't interrupt mid-conversation).
        """
        stripped = message.strip()

        # If there's any order-like content, not vague
        if re.search(r'\bORD[-\s]?\d', stripped, re.IGNORECASE):
            return False

        if not cls._VAGUE_PATTERNS.match(stripped):
            return False

        # If there's prior history with an order ID, the user is continuing
        # a conversation — don't ask for order ID again
        history = _session_history[session_id]
        if history:
            combined = " ".join(t["content"] for t in history)
            if re.search(r'\bORD-\d{5}\b', combined, re.IGNORECASE):
                return False

        return True

    # ------------------------------------------------------------------
    # Order ID format validation
    # ------------------------------------------------------------------
    _VALID_ORDER_ID   = re.compile(r'\bORD-\d{5}\b', re.IGNORECASE)
    _INVALID_ORDER_ID = re.compile(
        r'\b(ORD-\d{1,4}|ORD-\d{6,}|ORD-[A-Za-z]+|ORDER-\w+)\b',
        re.IGNORECASE,
    )

    @classmethod
    def _check_order_id_format(cls, message: str) -> str | None:
        """
        Returns a format-hint string if the message contains a malformed
        order ID, or None if everything looks fine.

        Does NOT fire for messages that contain no order ID at all —
        those are handled by the LLM or vague-help path above.
        """
        # Valid ID present → no problem
        if cls._VALID_ORDER_ID.search(message):
            return None

        match = cls._INVALID_ORDER_ID.search(message)
        if match:
            bad = match.group(0).upper()
            return (
                f"The order ID **{bad}** doesn't look quite right. "
                f"Order IDs follow the format **ORD-XXXXX** (5 digits), "
                f"for example **ORD-78321**. "
                f"You can find your order ID in your confirmation email. "
                f"Could you please double-check and try again?"
            )

        return None  # No malformed ID — pass through to LLM

    # ------------------------------------------------------------------
    # Auto-fill refund amount from order data
    # ------------------------------------------------------------------
    async def _auto_fill_refund_amount(
        self,
        plan: ActionPlan,
        customer_id: str,
    ) -> ActionPlan:
        """
        If a process_refund or escalate step is missing amount_inr,
        fetch the order and auto-fill the total of active items.

        This handles the case where a customer says:
        "I want to return order ORD-88001 and get a refund"
        without specifying the amount.
        """
        from agent.planner import ActionType
        from tools.oms_tool import OmsTool

        oms = OmsTool()

        for step in plan.steps:
            if step.action not in (ActionType.PROCESS_REFUND, ActionType.ESCALATE):
                continue
            if step.params.get("amount_inr") is not None:
                continue

            order_id = step.params.get("order_id")
            if not order_id:
                continue

            try:
                order = await oms.get_order(order_id.strip().upper())

                # Ownership check before using data
                if order.get("customer_id") != customer_id:
                    continue

                active_total = sum(
                    i["unit_price"] * i["quantity"]
                    for i in order.get("items", [])
                    if i.get("status") == "active"
                )
                step.params["amount_inr"] = round(active_total, 2)

                logger.info(
                    "Auto-filled refund amount | order=%s | amount=%.2f",
                    order_id, active_total,
                )
            except Exception as exc:
                # Silently pass — the tool will raise a proper error
                # when executed if the order doesn't exist
                logger.debug(
                    "Could not auto-fill refund amount | order=%s | err=%s",
                    order_id, exc,
                )

        return plan