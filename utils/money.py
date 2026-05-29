from decimal import Decimal, ROUND_HALF_UP
from typing import Union

_CENT = Decimal("0.01")


def to_decimal(amount: Union[int, float, str]) -> Decimal:
    return Decimal(str(amount)).quantize(_CENT, rounding=ROUND_HALF_UP)


def round_inr(amount: Union[int, float, str]) -> float:
    return float(to_decimal(amount))


def sum_active_items(items: list) -> float:
    total = sum(
        (Decimal(str(item["unit_price"])) * item["quantity"]
         for item in items if item.get("status") == "active"),
        Decimal("0"),
    )
    return float(total.quantize(_CENT, rounding=ROUND_HALF_UP))
