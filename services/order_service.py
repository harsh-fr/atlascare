import copy
import logging
from typing import Any

from utils.money import sum_active_items

logger = logging.getLogger(__name__)


class OrderService:
    def cancel_item(self, order: dict[str, Any], line_id: int) -> dict[str, Any]:
        updated = copy.deepcopy(order)
        target = next((i for i in updated["items"] if i["line_id"] == line_id), None)
        if target is None:
            raise ValueError(
                f"Line item {line_id} not found in order '{order.get('order_id')}'."
            )
        target["status"] = "cancelled"
        updated["total_amount"] = sum_active_items(updated["items"])
        logger.info(
            "OrderService.cancel_item | order=%s | line_id=%d | item=%s | new_total=%.2f",
            updated.get("order_id"), line_id, target.get("name"), updated["total_amount"],
        )
        return updated

    def update_shipping_address(
        self,
        order: dict[str, Any],
        new_address: dict[str, Any],
    ) -> dict[str, Any]:
        updated = copy.deepcopy(order)
        # Strip 'label' — it belongs to the CRM profile, not the order schema.
        shipping = {k: v for k, v in new_address.items() if k != "label"}
        updated["shipping_address"] = shipping
        logger.info(
            "OrderService.update_shipping_address | order=%s | city=%s | pincode=%s",
            updated.get("order_id"), shipping.get("city"), shipping.get("pincode"),
        )
        return updated

    def get_active_items(self, order: dict[str, Any]) -> list[dict[str, Any]]:
        return [i for i in order.get("items", []) if i.get("status") == "active"]

    def calculate_active_total(self, order: dict[str, Any]) -> float:
        return sum_active_items(order.get("items", []))

    def is_mutable(self, order: dict[str, Any]) -> bool:
        return order.get("status") not in {"delivered", "cancelled"}

    def can_cancel_items(self, order: dict[str, Any]) -> bool:
        return order.get("status") in {"placed", "processing"}
