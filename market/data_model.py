from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union


class Side(Enum):
    BUY = auto()
    SELL = auto()


class Order:
    pass


@dataclass(frozen=True)
class LimitOrder(Order):
    price: int
    volume: int
    side: Side
    type: str = field(default="L", init=False)


@dataclass(frozen=True)
class CancelOrder(Order):
    order_id: str
    type: str = field(default="C", init=False)


@dataclass(frozen=True)
class MarketOrder(Order):
    volume: int
    side: Side
    type: str = field(default="M", init=False)


@dataclass(frozen=True)
class OrderReceipt:
    outcome: bool
    order_id: Union[str, None]
    order_type: Union[str, None]
    average_price: Union[float, int, None]
    volume: Union[str, None]
    side: Union[Side, None]
