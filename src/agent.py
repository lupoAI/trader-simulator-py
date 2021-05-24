from enum import auto, Enum
import uuid


class Agent:

    def __init__(self, exchange):
        raise NotImplementedError

    def market_order(self, side):
        raise NotImplementedError

    def limit_order(self, side, price, volume):
        raise NotImplementedError

    def cancel_order(self, volume):
        raise NotImplementedError


class Exchange:

    def __init__(self):
        self.price_to_orders = {}
        self.orders_id_to_price = {}
        self.orders_id_to_agent = {}
        self.best_bid = 0
        self.best_ask = 0
        # TODO
        raise NotImplementedError

    def handle_order(self, order):
        raise NotImplementedError


class Order:

    def __init__(self):
        raise NotImplementedError


class Side(Enum):
    BUY = auto()
    SELL = auto()


order_id = str(uuid.uuid4())
