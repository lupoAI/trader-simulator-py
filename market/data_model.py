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

    def trade_volume(self, volume_traded):
        object.__setattr__(self, 'volume', self.volume - volume_traded)


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
    outcome: bool = False
    order_id: Union[str, None] = None
    order_type: Union[str, None] = None
    average_price: Union[float, int, None] = None
    volume: Union[int, None] = None
    side: Union[Side, None] = None
