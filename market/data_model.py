import uuid
from enum import Enum, auto


class Side(Enum):
    BUY = auto()
    SELL = auto()


class Order:
    pass


class LimitOrder(Order):

    def __init__(self, side, price, volume):
        self.type = "L"
        self.price = price
        self.side = side
        self.volume = volume


class CancelOrder(Order):

    def __init__(self, order_id):
        self.type = "C"
        self.order_id = order_id


class MarketOrder(Order):

    def __init__(self, side, volume):
        self.type = "M"
        self.side = side
        self.volume = volume


order_id = str(uuid.uuid4())
