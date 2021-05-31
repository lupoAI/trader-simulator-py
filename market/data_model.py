from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Union, List, Dict

from pandas import DataFrame


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


@dataclass
class Series:
    time_step: List[int] = field(default_factory=list)
    price: List[float] = field(default_factory=list)

    def add(self, time_step: int, price: float):
        self.time_step += [time_step]
        self.price += [price]

    def __getitem__(self, item):
        return self.price[item]

    def __len__(self):
        return len(self.price)



@dataclass(frozen=True)
class MarketSnapshot:
    time_step: int
    price_to_volume: Dict[int, int]
    best_bid_price: float
    best_ask_price: float
    best_bid_volume: int
    best_ask_volume: int
    total_bid_volume: int
    total_ask_volume: int
    mid_price: float


@dataclass
class MarketSnapshotSeries:
    time_step: List[int] = field(default_factory=list)
    price_to_volume: List[Dict[int, int]] = field(default_factory=list)
    best_bid_price: List[float] = field(default_factory=list)
    best_ask_price: List[float] = field(default_factory=list)
    best_bid_volume: List[int] = field(default_factory=list)
    best_ask_volume: List[int] = field(default_factory=list)
    total_bid_volume: List[int] = field(default_factory=list)
    total_ask_volume: List[int] = field(default_factory=list)
    mid_price: List[float] = field(default_factory=list)
    price_to_volume_df: DataFrame = field(default=DataFrame(), init=False)

    def add(self, other: MarketSnapshot):
        self.time_step += [other.time_step]
        self.price_to_volume += [other.price_to_volume]
        self.best_bid_price += [other.best_bid_price]
        self.best_ask_price += [other.best_ask_price]
        self.best_bid_volume += [other.best_bid_volume]
        self.best_ask_volume += [other.best_ask_volume]
        self.total_bid_volume += [other.total_bid_volume]
        self.total_ask_volume += [other.total_ask_volume]
        self.mid_price += [other.mid_price]

    def format_price_to_volume(self):
        self.price_to_volume_df = DataFrame(self.price_to_volume)
        self.price_to_volume_df.index = self.time_step
        self.price_to_volume_df.sort_index(axis=1, inplace=True)
