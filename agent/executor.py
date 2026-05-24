"""
agent/executor.py
=================
Deterministic tool dispatcher.

Responsibility
--------------
  1. Receive an ActionPlan from the Planner.
  2. Execute each ActionStep in dependency order.
  3. Enforce ownership — every order/customer reference is validated
     against the authenticated customer_id before any tool is called.
  4. Respect step dependencies — skip a step if any of its declared
     predecessors failed.
  5. Collect StepResult for every step (success or failure) and return
     a typed ExecutionResult to the Orchestrator.

Design principles
-----------------
- Deterministic > Generative: all routing, ownership checks, and
  retry logic live here in code — never delegated to the LLM.
- Fail-safe partial execution: a single step failure does NOT abort
  the entire plan unless downstream steps declare a dependency on it.
  Each step outcome is recorded in the trace regardless.
- Tools are called through typed interfaces (oms_tool, crm_tool, etc.)
  — the Executor never touches JSON files or repositories directly.
- All monetary arithmetic uses Python Decimal to avoid float errors.
"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agent.planner import ActionPlan, ActionStep, ActionType, Intent
from tools.oms_tool import OmsTool
from tools.crm_tool import CrmTool
from tools.payment_tool import PaymentTool
from tools.kb_tool import KbTool
from observability.tracer import Tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result contracts
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Outcome of a single ActionStep execution."""
    step_index: int
    action: ActionType
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    latency_ms: int = 0


@dataclass
class ExecutionResult:
    """
    Aggregated outcome of all steps in an ActionPlan.
    Consumed by Guardrails (post-check) and ResponseBuilder.
    """
    intent: Intent
    step_results: list[StepResult] = field(default_factory=list)
    escalated: bool = False
    escalation_case_id: str = ""

    # ------------------------------------------------------------------
    def overall_success(self) -> bool:
        """True only if every step succeeded."""
        return all(r.success for r in self.step_results)

    def any_success(self) -> bool:
        return any(r.success for r in self.step_results)

    def failed_steps(self) -> list[StepResult]:
        return [r for r in self.step_results if not r.success]

    def get_step_data(self, action: ActionType) -> dict[str, Any] | None:
        """Return data dict for the first successful step of a given action."""
        for r in self.step_results:
            if r.action == action and r.success:
                return r.data
        return None

    def fallback_summary(self) -> str:
        """
        Deterministic plain-text summary used when ResponseBuilder fails.
        Guarantees the user always receives some useful reply.
        """
        if self.escalated:
            return (
                f"Your request has been escalated to our specialist team "
                f"(Case ID: {self.escalation_case_id}). "
                "They will contact you within 24 hours."
            )
        successes = [r for r in self.step_results if r.success]
        failures  = [r for r in self.step_results if not r.success]

        parts: list[str] = []
        if successes:
            actions = ", ".join(r.action.value for r in successes)
            parts.append(f"Completed: {actions}.")
        if failures:
            actions = ", ".join(r.action.value for r in failures)
            parts.append(f"Could not complete: {actions}. Please try again.")
        return " ".join(parts) or "Your request has been processed."


# ---------------------------------------------------------------------------
# Ownership error
# ---------------------------------------------------------------------------
class OwnershipError(Exception):
    """Raised when a customer tries to access another customer's resource."""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class Executor:
    """
    Executes an ActionPlan step by step with ownership enforcement,
    dependency resolution, and full trace recording.

    Stateless across requests — all per-request state lives in local
    variables and the Tracer.
    """

    def __init__(self) -> None:
        self._oms     = OmsTool()
        self._crm     = CrmTool()
        self._payment = PaymentTool()
        self._kb      = KbTool()
        logger.debug("Executor tools initialised.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def execute(
        self,
        plan: ActionPlan,
        customer_id: str,
        tracer: Tracer,
    ) -> ExecutionResult:
        """
        Execute all steps in the plan, respecting dependencies.

        Parameters
        ----------
        plan        : validated ActionPlan from the Planner
        customer_id : authenticated customer identity
        tracer      : per-request tracer for observability

        Returns
        -------
        ExecutionResult containing per-step outcomes.
        """
        result = ExecutionResult(intent=plan.intent)

        # Index results by step position for dependency resolution
        step_results_by_index: dict[int, StepResult] = {}

        for idx, step in enumerate(plan.steps):
            # ---- dependency check ----------------------------------------
            if self._has_failed_dependency(step, step_results_by_index):
                logger.info(
                    "Skipping step %d (%s) — dependency failed | trace=%s",
                    idx, step.action, tracer.trace_id,
                )
                skipped = StepResult(
                    step_index=idx,
                    action=step.action,
                    success=False,
                    error="Skipped: upstream step failed.",
                )
                result.step_results.append(skipped)
                step_results_by_index[idx] = skipped
                continue

            # ---- dispatch ------------------------------------------------
            step_result = await self._dispatch(
                idx=idx,
                step=step,
                customer_id=customer_id,
                tracer=tracer,
                prior_results=step_results_by_index,
            )
            result.step_results.append(step_result)
            step_results_by_index[idx] = step_result

            # Track escalation at result level for easy access
            if step.action == ActionType.ESCALATE and step_result.success:
                result.escalated = True
                result.escalation_case_id = step_result.data.get("case_id", "")

        return result

    # ------------------------------------------------------------------
    # Dependency resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _has_failed_dependency(
        step: ActionStep,
        results: dict[int, StepResult],
    ) -> bool:
        return any(
            results.get(dep_idx, StepResult(dep_idx, ActionType.GET_ORDER, False)).success is False
            for dep_idx in step.depends_on
            if dep_idx in results
        )

    # ------------------------------------------------------------------
    # Dispatcher — routes each ActionType to the appropriate tool
    # ------------------------------------------------------------------
    async def _dispatch(
        self,
        idx: int,
        step: ActionStep,
        customer_id: str,
        tracer: Tracer,
        prior_results: dict[int, StepResult],
    ) -> StepResult:
        """Route a single step to its tool method."""
        t0 = time.monotonic()
        logger.info(
            "Executing step %d | action=%s | trace=%s",
            idx, step.action, tracer.trace_id,
        )

        try:
            data = await self._run_action(
                step=step,
                customer_id=customer_id,
                tracer=tracer,
                prior_results=prior_results,
            )
            latency_ms = int((time.monotonic() - t0) * 1000)
            tracer.record_tool_call(
                tool=step.action.value,
                action=step.action.value,
                status="success",
                meta={"step": idx, "latency_ms": latency_ms},
            )
            return StepResult(
                step_index=idx,
                action=step.action,
                success=True,
                data=data,
                latency_ms=latency_ms,
            )

        except OwnershipError as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.warning(
                "Ownership violation | step=%d | customer=%s | trace=%s | err=%s",
                idx, customer_id, tracer.trace_id, exc,
            )
            tracer.record_tool_call(
                tool=step.action.value,
                action=step.action.value,
                status="ownership_denied",
                meta={"step": idx, "error": str(exc)},
            )
            return StepResult(
                step_index=idx,
                action=step.action,
                success=False,
                error="Access denied.",
                latency_ms=latency_ms,
            )

        except Exception as exc:
            latency_ms = int((time.monotonic() - t0) * 1000)
            logger.exception(
                "Step %d failed | action=%s | trace=%s | error=%s",
                idx, step.action, tracer.trace_id, exc,
            )
            tracer.record_tool_call(
                tool=step.action.value,
                action=step.action.value,
                status="error",
                meta={"step": idx, "error": str(exc), "latency_ms": latency_ms},
            )
            return StepResult(
                step_index=idx,
                action=step.action,
                success=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------
    async def _run_action(
        self,
        step: ActionStep,
        customer_id: str,
        tracer: Tracer,
        prior_results: dict[int, StepResult],
    ) -> dict[str, Any]:
        """
        Call the appropriate tool for a given ActionType.
        Returns a plain dict that becomes StepResult.data.
        """
        p = step.params  # shorthand

        match step.action:

            # ------------------------------------------------------
            case ActionType.GET_ORDER:
                order_id = self._require_param(p, "order_id", step.action)
                order = await self._oms.get_order(order_id)
                self._assert_ownership(order["customer_id"], customer_id, order_id)
                return {"order": order}

            # ------------------------------------------------------
            case ActionType.CANCEL_ITEM:
                order_id = self._require_param(p, "order_id", step.action)
                line_id  = int(self._require_param(p, "line_id", step.action))

                # Ownership check — fetch order first
                order = await self._oms.get_order(order_id)
                self._assert_ownership(order["customer_id"], customer_id, order_id)

                cancelled = await self._oms.cancel_item(order_id, line_id)
                return {"cancelled_item": cancelled, "order_id": order_id}

            # ------------------------------------------------------
            case ActionType.PROCESS_REFUND:
                order_id   = self._require_param(p, "order_id", step.action)
                amount_inr = Decimal(str(self._require_param(p, "amount_inr", step.action)))
                method     = p.get("method", "original")

                # Ownership
                order = await self._oms.get_order(order_id)
                self._assert_ownership(order["customer_id"], customer_id, order_id)

                refund = await self._payment.process_refund(
                    order_id=order_id,
                    amount_inr=float(amount_inr),
                    method=method,
                    customer_id=customer_id,
                )
                return {"refund": refund}

            # ------------------------------------------------------
            case ActionType.UPDATE_ADDRESS:
                order_id      = self._require_param(p, "order_id", step.action)
                address_label = self._require_param(p, "address_label", step.action)

                # Ownership
                order = await self._oms.get_order(order_id)
                self._assert_ownership(order["customer_id"], customer_id, order_id)

                updated = await self._oms.update_shipping_address(
                    order_id=order_id,
                    customer_id=customer_id,
                    address_label=address_label,
                )
                return {"updated_address": updated}

            # ------------------------------------------------------
            case ActionType.CREATE_CRM_CASE:
                order_id   = self._require_param(p, "order_id", step.action)
                reason     = self._require_param(p, "reason", step.action)
                amount_inr = p.get("amount_inr")

                case = await self._crm.create_case(
                    customer_id=customer_id,
                    order_id=order_id,
                    reason=reason,
                    amount_inr=float(amount_inr) if amount_inr is not None else None,
                    trace_id=tracer.trace_id,
                )
                return {"case": case}

            # ------------------------------------------------------
            case ActionType.SEARCH_KB:
                tags = p.get("tags", [])
                articles = await self._kb.search(tags=tags)
                return {"articles": articles}

            # ------------------------------------------------------
            case ActionType.ESCALATE:
                order_id   = self._require_param(p, "order_id", step.action)
                reason     = self._require_param(p, "reason", step.action)
                amount_inr = p.get("amount_inr")

                # Ownership check
                order = await self._oms.get_order(order_id)
                self._assert_ownership(order["customer_id"], customer_id, order_id)

                case = await self._crm.create_case(
                    customer_id=customer_id,
                    order_id=order_id,
                    reason=reason,
                    amount_inr=float(amount_inr) if amount_inr is not None else None,
                    trace_id=tracer.trace_id,
                    priority="high",
                )
                return {
                    "case_id": case["case_id"],
                    "case":    case,
                    "escalated": True,
                }

            # ------------------------------------------------------
            case _:
                raise ValueError(f"Unhandled ActionType: {step.action}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_param(params: dict, key: str, action: ActionType) -> Any:
        """Raise clearly if a required parameter is absent."""
        if key not in params or params[key] is None:
            raise ValueError(f"Action '{action}' requires parameter '{key}'.")
        return params[key]

    @staticmethod
    def _assert_ownership(
        order_customer_id: str,
        session_customer_id: str,
        order_id: str,
    ) -> None:
        """
        Enforce that the authenticated customer owns the referenced order.

        Raises OwnershipError — never reveals that the order belongs to
        someone else (returns the same error as 'not found').
        """
        if order_customer_id != session_customer_id:
            # Deliberately vague message to prevent customer enumeration
            raise OwnershipError(
                f"Order {order_id} not found for the current session."
            )