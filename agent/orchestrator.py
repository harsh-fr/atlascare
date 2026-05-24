"""
agent/orchestrator.py
=====================
Central coordinator for AtlasCare.

Responsibility chain
--------------------
  1. Resolve session → customer identity  (SessionStore)
  2. Run pre-execution guardrails          (Guardrails)
  3. Ask the Planner to produce an action plan (LLM-assisted)
  4. Hand the plan to the Executor for tool-by-tool execution
  5. Pass execution results to the ResponseBuilder for final phrasing
  6. Return a structured OrchestratorResult to main.py

The Orchestrator owns the *flow*.
It does NOT own business logic, tool calls, or LLM prompts —
those belong to their respective layers.

Design principles applied here
--------------------------------
- Deterministic > Generative: guardrails and ownership checks
  happen in code BEFORE the LLM sees the request.
- Fail-safe: any unhandled exception inside executor/planner is
  caught here, logged with trace context, and returned as a safe
  user-facing message — never a 500 to the end user.
- Single responsibility: orchestrator wires components; it doesn't
  implement them.
"""

import logging
import re
from dataclasses import dataclass

from agent.planner import Planner, ActionPlan
from agent.executor import Executor, ExecutionResult
from agent.guardrails import Guardrails, GuardrailVerdict
from agent.response_builder import ResponseBuilder
from observability.tracer import Tracer
from utils.session_store import SessionStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result contract returned to main.py
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorResult:
    """Thin result envelope returned to the HTTP layer."""
    response_text: str


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """
    Wires together the full agent pipeline for a single request.

    Instantiated once at application startup (app.state.orchestrator)
    and reused across requests — all per-request state lives in the
    Tracer and local variables, not in instance attributes.
    """

    def __init__(self) -> None:
        self._session_store = SessionStore()
        self._guardrails = Guardrails()
        self._planner = Planner()
        self._executor = Executor()
        self._response_builder = ResponseBuilder()
        logger.debug("Orchestrator components initialised.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _check_order_id_format(message: str) -> str | None:
        """
        Detect if the message contains something that looks like an
        order ID attempt but doesn't match the valid ORD-XXXXX pattern.

        Returns a helpful user-facing correction message if an invalid
        format is detected, or None if everything looks fine.

        Valid format   : ORD-78321  (ORD- followed by exactly 5 digits)
        Invalid examples: ORD-123, ORD-ABCDE, ORDER-78321, 78321
        """
        # Check if there's a valid order ID already — if so, no problem
        valid_pattern   = re.compile(r'\bORD-\d{5}\b', re.IGNORECASE)
        # Detect malformed attempts: ORD- with wrong suffix, or ORDER-
        invalid_pattern = re.compile(
            r'\b(ORD-\d{1,4}\b|ORD-\d{6,}\b|ORD-[A-Z]+\b|ORDER-\w+\b)',
            re.IGNORECASE,
        )

        if valid_pattern.search(message):
            return None  # Valid order ID present — no issue

        match = invalid_pattern.search(message)
        if match:
            bad_id = match.group(0).upper()   # normalise for display
            return (
                f"It looks like the order ID **{bad_id}** may not be in the "
                f"correct format. Order IDs should follow the pattern "
                f"**ORD-XXXXX** (5 digits), for example **ORD-78321**. "
                f"Could you please check your order ID and try again? "
                f"You can find it in your order confirmation email."
            )

        return None  # No order ID attempt detected — let LLM handle it
    async def handle(
        self,
        message: str,
        session_id: str,
        tracer: Tracer,
    ) -> OrchestratorResult:
        """
        Execute the full agent pipeline for one customer request.

        Parameters
        ----------
        message    : raw customer message
        session_id : opaque session token from the HTTP request
        tracer     : per-request Tracer; populated throughout and
                     read back by main.py to build the trace payload

        Returns
        -------
        OrchestratorResult with the final user-facing response text.
        """

        # ----------------------------------------------------------
        # Step 1 — Resolve customer identity from session
        # ----------------------------------------------------------
        customer_id = self._session_store.resolve(session_id)
        if customer_id is None:
            logger.warning(
                "Unresolvable session | trace=%s | session=%s",
                tracer.trace_id,
                session_id,
            )
            return OrchestratorResult(
                response_text=(
                    "I'm unable to verify your session. "
                    "Please log in again and retry."
                )
            )

        tracer.set_customer_id(customer_id)
        logger.info(
            "Session resolved | trace=%s | customer=%s",
            tracer.trace_id,
            customer_id,
        )

        # ----------------------------------------------------------
        # Step 2 — Pre-execution guardrail check (deterministic)
        #
        # Guardrails run BEFORE the LLM so that policy violations
        # are blocked in code, not delegated to prompt logic.
        # ----------------------------------------------------------
        verdict: GuardrailVerdict = self._guardrails.pre_check(
            message=message,
            customer_id=customer_id,
            tracer=tracer,
        )
        if verdict.blocked:
            logger.warning(
                "Request blocked by pre-guardrail | trace=%s | reason=%s",
                tracer.trace_id,
                verdict.reason,
            )
            return OrchestratorResult(response_text=verdict.user_message)

        # ----------------------------------------------------------
        # Step 2b — Order ID format check (deterministic, pre-LLM)
        #
        # If the message contains something that looks like an order
        # ID but doesn't match ORD-XXXXX, catch it immediately and
        # return a helpful format hint without wasting an LLM call.
        # ----------------------------------------------------------
        order_id_hint = self._check_order_id_format(message)
        if order_id_hint:
            return OrchestratorResult(response_text=order_id_hint)

        # ----------------------------------------------------------
        # Step 3 — Planning (LLM-assisted intent → action plan)
        # ----------------------------------------------------------
        try:
            plan: ActionPlan = await self._planner.plan(
                message=message,
                customer_id=customer_id,
                tracer=tracer,
            )
        except Exception as exc:
            logger.exception(
                "Planner failure | trace=%s | error=%s",
                tracer.trace_id,
                exc,
            )
            return OrchestratorResult(
                response_text=(
                    "I encountered an issue understanding your request. "
                    "Please rephrase and try again."
                )
            )

        logger.info(
            "Plan produced | trace=%s | intent=%s | steps=%d",
            tracer.trace_id,
            plan.intent,
            len(plan.steps),
        )

        # ----------------------------------------------------------
        # Step 4 — Execution (deterministic tool dispatch)
        # ----------------------------------------------------------
        try:
            execution_result: ExecutionResult = await self._executor.execute(
                plan=plan,
                customer_id=customer_id,
                tracer=tracer,
            )
        except Exception as exc:
            logger.exception(
                "Executor failure | trace=%s | error=%s",
                tracer.trace_id,
                exc,
            )
            return OrchestratorResult(
                response_text=(
                    "Something went wrong while processing your request. "
                    "Our team has been notified. Please try again shortly."
                )
            )

        # ----------------------------------------------------------
        # Step 5 — Post-execution guardrail check
        #
        # Second safety gate: verify execution outcomes satisfy
        # policy (e.g. confirm no autonomous payment was made for
        # an escalation case).
        # ----------------------------------------------------------
        post_verdict: GuardrailVerdict = self._guardrails.post_check(
            execution_result=execution_result,
            tracer=tracer,
        )
        if post_verdict.blocked:
            logger.error(
                "Post-execution guardrail triggered | trace=%s | reason=%s",
                tracer.trace_id,
                post_verdict.reason,
            )
            # Surface as a safe holding message; the trace preserves details.
            return OrchestratorResult(response_text=post_verdict.user_message)

        # ----------------------------------------------------------
        # Step 6 — Response assembly (LLM-assisted phrasing)
        # ----------------------------------------------------------
        try:
            response_text = await self._response_builder.build(
                message=message,
                plan=plan,
                execution_result=execution_result,
                customer_id=customer_id,
                tracer=tracer,
            )
        except Exception as exc:
            logger.exception(
                "ResponseBuilder failure | trace=%s | error=%s",
                tracer.trace_id,
                exc,
            )
            # Fall back to a deterministic summary so the user is never
            # left without a reply even if the LLM phrasing step fails.
            response_text = execution_result.fallback_summary()

        return OrchestratorResult(response_text=response_text)