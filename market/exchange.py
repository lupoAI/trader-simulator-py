from collections import deque
import uuid
from numpy import inf
from bisect import insort
from market.data_model import Order
from market.data_model import Side
from market.data_model import LimitOrder, MarketOrder, CancelOrder


class Exchange:

    def __init__(self):
        self.agent_id_to_agent = {}
        self.order_id_to_order = {}
        self.order_id_to_agent_id = {}
        self.price_to_orders_ids = {}

        self.sorted_bid_prices = []
        self.sorted_ask_prices = []

        self.best_bid_price = 0
        self.best_ask_price = inf
        self.best_bid_volume = 0
        self.best_ask_volume = 0

        self.true_price = 10000

    def subscribed(self, agent_self):
        agent_id = str(uuid.uuid4())
        self.agent_id_to_agent[agent_id] = agent_self
        return agent_id

    def handle_order(self, order: Order, agent_id):
        if order.type == "L":
            order_id = str(uuid.uuid4())
            self.handle_limit_order(order, order_id)
            self.order_id_to_agent_id[order_id] = agent_id
            self.order_id_to_order[order_id] = order
            return order_id
        elif order.type == "M":
            order_id = str(uuid.uuid4())
            #TODO
            raise NotImplementedError
        elif order.type == "C":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def handle_limit_order(self, order: LimitOrder, order_id: str):
        market_order_buy = order.side == Side.BUY \
                           and order.price >= self.best_ask_price
        market_order_sell = order.side == Side.SELL \
                            and order.price <= self.best_bid_price

        if market_order_buy or market_order_sell:
            mo = MarketOrder(order.side, order.volume)
            self.handle_market_order(mo, order_id)
        elif order.price not in self.price_to_orders_ids:
            self.price_to_orders_ids[order.price] = deque([order_id])
            if order.side == Side.BUY:
                insort(self.sorted_bid_prices, order.price)
                self.best_bid_price = self.sorted_bid_prices[-1]
            else:
                insort(self.sorted_ask_prices, order.price)
                self.best_ask_price = self.sorted_ask_prices[0]
        else:
            self.price_to_orders_ids[order.price].append([order_id])


        def handle_market_order(self, order: MarketOrder, order_id: str):
            #TODO
            raise NotImplementedError

        def handle_cancel_order(self, order_id):
            #TODO
            raise NotImplementedError
