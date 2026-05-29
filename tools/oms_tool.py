import logging
from typing import Any

from repositories.order_repository import OrderRepository
from services.order_service import OrderService

logger = logging.getLogger(__name__)


class OmsError(Exception):
    pass

class OrderNotFoundError(OmsError):
    pass

class LineItemNotFoundError(OmsError):
    pass

class ItemAlreadyCancelledError(OmsError):
    pass

class OrderNotMutableError(OmsError):
    pass

class AddressNotFoundError(OmsError):
    pass


class OmsTool:
    def __init__(self) -> None:
        self._order_repo = OrderRepository()
        self._order_svc  = OrderService()
        logger.debug("OmsTool initialised.")

    async def get_order(self, order_id: str) -> dict[str, Any]:
        logger.debug("OmsTool.get_order | order_id=%s", order_id)
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order '{order_id}' not found.")
        return order

    async def cancel_item(self, order_id: str, line_id: int) -> dict[str, Any]:
        logger.info("OmsTool.cancel_item | order_id=%s | line_id=%d", order_id, line_id)
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order '{order_id}' not found.")

        mutable_statuses = {"placed", "processing"}
        if order["status"] not in mutable_statuses:
            raise OrderNotMutableError(
                f"Order '{order_id}' has status '{order['status']}' and "
                f"cannot be partially cancelled. "
                f"Only orders in {mutable_statuses} are eligible."
            )

        item = next((i for i in order["items"] if i["line_id"] == line_id), None)
        if item is None:
            raise LineItemNotFoundError(f"Line item {line_id} not found in order '{order_id}'.")

        if item["status"] == "cancelled":
            raise ItemAlreadyCancelledError(
                f"Line item {line_id} in order '{order_id}' is already cancelled."
            )

        updated_order = self._order_svc.cancel_item(order=order, line_id=line_id)
        self._order_repo.save(updated_order)

        cancelled_item = next(i for i in updated_order["items"] if i["line_id"] == line_id)
        logger.info(
            "OmsTool.cancel_item SUCCESS | order_id=%s | line_id=%d | item=%s | amount=%.2f",
            order_id, line_id, cancelled_item["name"], cancelled_item["unit_price"],
        )
        return {
            "order_id":        order_id,
            "line_id":         line_id,
            "name":            cancelled_item["name"],
            "unit_price":      cancelled_item["unit_price"],
            "quantity":        cancelled_item["quantity"],
            "status":          "cancelled",
            "new_order_total": updated_order["total_amount"],
        }

    async def list_orders(self, customer_id: str) -> list[dict[str, Any]]:
        logger.debug("OmsTool.list_orders | customer_id=%s", customer_id)
        return self._order_repo.find_by_customer(customer_id)

    async def update_shipping_address(
        self,
        order_id: str,
        customer_id: str,
        address_label: str,
    ) -> dict[str, Any]:
        logger.info(
            "OmsTool.update_shipping_address | order_id=%s | customer_id=%s | label=%s",
            order_id, customer_id, address_label,
        )
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order '{order_id}' not found.")

        immutable_statuses = {"shipped", "delivered", "cancelled"}
        if order["status"] in immutable_statuses:
            raise OrderNotMutableError(
                f"Cannot update address for order '{order_id}' "
                f"with status '{order['status']}' — the package is already in transit."
            )

        from repositories.crm_repository import CrmRepository
        customer = CrmRepository().find_customer_by_id(customer_id)
        if customer is None:
            raise AddressNotFoundError(f"Customer profile not found for '{customer_id}'.")

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

        current = order.get("shipping_address", {})
        new = {k: target_address.get(k) for k in ("line1", "city", "state", "pincode")}
        if (current.get("line1") == new["line1"] and
                current.get("pincode") == new["pincode"]):
            logger.info(
                "OmsTool.update_shipping_address NO-OP | order_id=%s | label=%s | already set",
                order_id, address_label,
            )
            return {
                "order_id":      order_id,
                "address_label": address_label,
                "already_current": True,
                "current_address": current,
            }

        updated_order = self._order_svc.update_shipping_address(
            order=order, new_address=target_address,
        )
        self._order_repo.save(updated_order)
        logger.info(
            "OmsTool.update_shipping_address SUCCESS | order_id=%s | new_city=%s",
            order_id, target_address.get("city"),
        )
        return {
            "order_id":      order_id,
            "address_label": address_label,
            "new_address":   new,
        }

    async def update_shipping_address_raw(
        self,
        order_id: str,
        line1: str,
        city: str,
        state: str,
        pincode: str,
    ) -> dict[str, Any]:
        logger.info(
            "OmsTool.update_shipping_address_raw | order_id=%s | city=%s",
            order_id, city,
        )
        order = self._order_repo.find_by_id(order_id)
        if order is None:
            raise OrderNotFoundError(f"Order '{order_id}' not found.")

        immutable_statuses = {"shipped", "delivered", "cancelled"}
        if order["status"] in immutable_statuses:
            raise OrderNotMutableError(
                f"Cannot update address for order '{order_id}' "
                f"with status '{order['status']}' — the package is already in transit."
            )

        new_address = {"line1": line1, "city": city, "state": state, "pincode": pincode}
        updated_order = self._order_svc.update_shipping_address(
            order=order, new_address=new_address,
        )
        self._order_repo.save(updated_order)
        logger.info(
            "OmsTool.update_shipping_address_raw SUCCESS | order_id=%s | city=%s",
            order_id, city,
        )
        return {"order_id": order_id, "new_address": new_address}
