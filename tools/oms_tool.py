"""
tools/oms_tool.py
=================
Order Management System (OMS) integration tool.

Responsibility
--------------
  Exposes typed, async methods that the Executor calls to interact
  with order data:
    - get_order()               : fetch a single order by ID
    - cancel_item()             : partially cancel a line item
    - update_shipping_address() : update delivery address for active items

Design principles
-----------------
- Tools are the ONLY layer that touches repositories.
  Agent code never accesses repositories directly.
- Business rule validation (ownership, cancellability) lives in
  services/, not here. The tool is an integration adapter.
- Each method raises a typed OmsError so callers can handle
  OMS failures distinctly from programming errors.
- All mutations are validated for preconditions before being
  applied (item exists, item is active, order is mutable).
- Returns plain dicts (JSON-serialisable) — no ORM objects leak
  out of this layer.
"""

import logging
from typing import Any

from repositories.order_repository import OrderRepository
from services.order_service import OrderService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class OmsError(Exception):
    """Base error for all OMS tool failures."""

class OrderNotFoundError(OmsError):
    """Order ID does not exist in the repository."""

class LineItemNotFoundError(OmsError):
    """Line item ID does not exist in the order."""

class ItemAlreadyCancelledError(OmsError):
    """Line item is already in cancelled state."""

class OrderNotMutableError(OmsError):
    """Order status does not permit the requested mutation."""

class AddressNotFoundError(OmsError):
    """Requested address label not found in customer profile."""


# ---------------------------------------------------------------------------
# OmsTool
# ---------------------------------------------------------------------------
class OmsTool:
    """
    Typed async interface to the Order Management System.

    Backed today by JSON repositories; interface is stable so the
    backing store can be swapped to a REST API without changing callers.
    """

    def __init__(self) -> None:
        self._order_repo = OrderRepository()
        self._order_svc  = OrderService()
        logger.debug("OmsTool initialised.")

    # ------------------------------------------------------------------
    # get_order
    # ------------------------------------------------------------------
    async def get_order(self, order_id: str) -> dict[str, Any]:
        """
        Fetch a single order by ID.

        Order ID lookup is case-insensitive — "ord-78321" and
        "ORD-78321" resolve to the same order.

        Returns
        -------
        dict representation of the order (matches orders.json schema).

        Raises
        ------
        OrderNotFoundError  if order_id does not exist.
        OmsError            on any unexpected repository failure.
        """
        order_id = order_id.strip().upper()
        logger.debug("OmsTool.get_order | order_id=%s", order_id)

        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(
                f"Order '{order_id}' not found."
            )

        return order

    # ------------------------------------------------------------------
    # cancel_item
    # ------------------------------------------------------------------
    async def cancel_item(
        self,
        order_id: str,
        line_id: int,
    ) -> dict[str, Any]:
        """
        Partially cancel a single line item within an order.

        Business rules enforced
        -----------------------
        - Order must be in a mutable state (placed / processing).
          Shipped or delivered orders cannot be line-cancelled here;
          they require a return flow.
        - Line item must exist and must currently be active.

        Returns
        -------
        dict with keys: order_id, line_id, name, unit_price, status="cancelled"

        Raises
        ------
        OrderNotFoundError       if order_id does not exist.
        LineItemNotFoundError    if line_id does not exist in the order.
        ItemAlreadyCancelledError if the item is already cancelled.
        OrderNotMutableError     if order status prevents cancellation.
        """
        logger.info(
            "OmsTool.cancel_item | order_id=%s | line_id=%d",
            order_id,
            line_id,
        )

        order_id = order_id.strip().upper()
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order '{order_id}' not found.")

        # Order-level mutability check
        mutable_statuses = {"placed", "processing"}
        if order["status"] not in mutable_statuses:
            raise OrderNotMutableError(
                f"Order '{order_id}' has status '{order['status']}' and "
                f"cannot be partially cancelled. "
                f"Only orders in {mutable_statuses} are eligible."
            )

        # Locate the line item
        item = next(
            (i for i in order["items"] if i["line_id"] == line_id),
            None,
        )
        if item is None:
            raise LineItemNotFoundError(
                f"Line item {line_id} not found in order '{order_id}'."
            )

        if item["status"] == "cancelled":
            raise ItemAlreadyCancelledError(
                f"Line item {line_id} in order '{order_id}' is already cancelled."
            )

        # Apply cancellation via service (recalculates total_amount)
        updated_order = self._order_svc.cancel_item(
            order=order,
            line_id=line_id,
        )
        self._order_repo.save(updated_order)

        cancelled_item = next(
            i for i in updated_order["items"] if i["line_id"] == line_id
        )

        logger.info(
            "OmsTool.cancel_item SUCCESS | order_id=%s | line_id=%d | "
            "item=%s | amount=%.2f",
            order_id,
            line_id,
            cancelled_item["name"],
            cancelled_item["unit_price"],
        )

        return {
            "order_id":   order_id,
            "line_id":    line_id,
            "name":       cancelled_item["name"],
            "unit_price": cancelled_item["unit_price"],
            "quantity":   cancelled_item["quantity"],
            "status":     "cancelled",
            "new_order_total": updated_order["total_amount"],
        }

    # ------------------------------------------------------------------
    # update_shipping_address
    # ------------------------------------------------------------------
    async def update_shipping_address(
        self,
        order_id: str,
        customer_id: str,
        address_label: str,
    ) -> dict[str, Any]:
        """
        Update the shipping address for an order to a saved address
        identified by label (e.g. "office", "home").

        Business rules enforced
        -----------------------
        - Order must not be delivered or cancelled.
        - Address label must exist in the customer's saved addresses
          (looked up from CRM repository).

        Returns
        -------
        dict with keys: order_id, address_label, new_address (dict)

        Raises
        ------
        OrderNotFoundError    if order_id does not exist.
        OrderNotMutableError  if order cannot have address changed.
        AddressNotFoundError  if address_label not in customer profile.
        """
        logger.info(
            "OmsTool.update_shipping_address | order_id=%s | "
            "customer_id=%s | label=%s",
            order_id,
            customer_id,
            address_label,
        )

        order_id = order_id.strip().upper()
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order '{order_id}' not found.")

        # Address changes not allowed on delivered/cancelled orders
        immutable_statuses = {"delivered", "cancelled"}
        if order["status"] in immutable_statuses:
            raise OrderNotMutableError(
                f"Cannot update address for order '{order_id}' "
                f"with status '{order['status']}'."
            )

        # Resolve the address from the CRM repository
        from repositories.crm_repository import CrmRepository
        crm_repo = CrmRepository()
        customer = crm_repo.find_customer_by_id(customer_id)
        if customer is None:
            raise AddressNotFoundError(
                f"Customer profile not found for '{customer_id}'."
            )

        addresses = customer.get("addresses", [])
        target_address = next(
            (a for a in addresses if a.get("label", "").lower() == address_label.lower()),
            None,
        )
        if target_address is None:
            available = [a.get("label") for a in addresses]
            raise AddressNotFoundError(
                f"Address label '{address_label}' not found for customer "
                f"'{customer_id}'. Available labels: {available}."
            )

        # Apply update via service
        updated_order = self._order_svc.update_shipping_address(
            order=order,
            new_address=target_address,
        )
        self._order_repo.save(updated_order)

        logger.info(
            "OmsTool.update_shipping_address SUCCESS | order_id=%s | "
            "new_city=%s",
            order_id,
            target_address.get("city"),
        )

        return {
            "order_id":      order_id,
            "address_label": address_label,
            "new_address":   {
                "line1":   target_address.get("line1"),
                "city":    target_address.get("city"),
                "state":   target_address.get("state"),
                "pincode": target_address.get("pincode"),
            },
        }