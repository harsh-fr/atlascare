"""
services/order_service.py
==========================
Order business logic layer.

Responsibility
--------------
  Owns all business rules and calculations related to orders:
    - cancel_item()              : mark a line item cancelled and
                                   recalculate order total
    - update_shipping_address()  : apply a new address to an order

Design principles
-----------------
- Pure business logic only — no I/O, no repository access.
  The tool layer owns I/O; this layer owns rules and arithmetic.
- All monetary arithmetic uses Decimal to avoid float precision errors.
- Returns a fully updated order dict — the caller (tool) persists it.
- Methods are stateless — safe to call concurrently.
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
import copy

logger = logging.getLogger(__name__)


class OrderService:
    """
    Stateless business logic for order mutations.

    Accepts and returns plain dicts matching the orders.json schema.
    Never touches repositories or tools directly.
    """

    # ------------------------------------------------------------------
    # cancel_item
    # ------------------------------------------------------------------
    def cancel_item(
        self,
        order: dict[str, Any],
        line_id: int,
    ) -> dict[str, Any]:
        """
        Mark a line item as cancelled and recalculate the order total.

        Parameters
        ----------
        order   : full order dict (will not be mutated — deep copy made)
        line_id : 1-based line item identifier

        Returns
        -------
        Updated order dict with:
          - items[line_id-1].status = "cancelled"
          - total_amount recalculated from remaining active items

        Raises
        ------
        ValueError  if line_id not found in order items.
        """
        updated = copy.deepcopy(order)

        # Locate and cancel the item
        target = next(
            (i for i in updated["items"] if i["line_id"] == line_id),
            None,
        )
        if target is None:
            raise ValueError(
                f"Line item {line_id} not found in order "
                f"'{order.get('order_id')}'."
            )

        target["status"] = "cancelled"

        # Recalculate total from active items using Decimal
        # start=Decimal("0") prevents sum() returning int(0) when no items remain
        new_total = sum(
            (Decimal(str(item["unit_price"])) * item["quantity"]
             for item in updated["items"]
             if item["status"] == "active"),
            Decimal("0"),
        )
        updated["total_amount"] = float(
            new_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

        logger.info(
            "OrderService.cancel_item | order=%s | line_id=%d | "
            "item=%s | new_total=%.2f",
            updated.get("order_id"),
            line_id,
            target.get("name"),
            updated["total_amount"],
        )

        return updated

    # ------------------------------------------------------------------
    # update_shipping_address
    # ------------------------------------------------------------------
    def update_shipping_address(
        self,
        order: dict[str, Any],
        new_address: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Replace the shipping address on an order.

        Parameters
        ----------
        order       : full order dict (deep copy made — original not mutated)
        new_address : address dict with keys: line1, city, state, pincode
                      (label key is stripped — not part of order schema)

        Returns
        -------
        Updated order dict with shipping_address replaced.
        """
        updated = copy.deepcopy(order)

        # Strip the 'label' key — it belongs to CRM, not the order schema
        shipping = {
            k: v for k, v in new_address.items()
            if k != "label"
        }

        updated["shipping_address"] = shipping

        logger.info(
            "OrderService.update_shipping_address | order=%s | "
            "city=%s | pincode=%s",
            updated.get("order_id"),
            shipping.get("city"),
            shipping.get("pincode"),
        )

        return updated

    # ------------------------------------------------------------------
    # Helpers / read-only queries
    # ------------------------------------------------------------------
    def get_active_items(
        self,
        order: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return only the active (non-cancelled) line items."""
        return [
            i for i in order.get("items", [])
            if i.get("status") == "active"
        ]

    def calculate_active_total(
        self,
        order: dict[str, Any],
    ) -> float:
        """
        Deterministically calculate total from active items.
        Used for refund amount validation and order summary.
        """
        total = sum(
            (Decimal(str(item["unit_price"])) * item["quantity"]
             for item in order.get("items", [])
             if item.get("status") == "active"),
            Decimal("0"),
        )
        return float(
            total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    def is_mutable(self, order: dict[str, Any]) -> bool:
        """
        Return True if the order can be modified (cancelled / address changed).
        Delivered and cancelled orders are immutable.
        """
        return order.get("status") not in {"delivered", "cancelled"}

    def can_cancel_items(self, order: dict[str, Any]) -> bool:
        """
        Return True if line-item cancellation is permitted.
        Only placed and processing orders support partial cancellation.
        """
        return order.get("status") in {"placed", "processing"}