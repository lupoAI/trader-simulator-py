import uuid
from bisect import insort
from collections import deque

from numpy import inf

from market.data_model import OrderReceipt
from market.data_model import LimitOrder, MarketOrder, CancelOrder
from market.data_model import Side
from typing import Union


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

        self.mid_price = 0
        self.true_price = 10000

    def subscribed(self, agent_self):
        agent_id = str(uuid.uuid4())
        self.agent_id_to_agent[agent_id] = agent_self
        return agent_id

    def handle_order(self, agent_id: str, order: Union[LimitOrder, MarketOrder, CancelOrder]):
        if order.type == "L":
            return self.handle_limit_order(agent_id, order)
        elif order.type == "M":
            order_id = str(uuid.uuid4())
            # TODO
            raise NotImplementedError
        elif order.type == "C":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def handle_limit_order(self, agent_id: str, order: LimitOrder):
        order_id = str(uuid.uuid4())

        market_order_buy = order.side == Side.BUY and order.price >= self.best_ask_price
        market_order_sell = order.side == Side.SELL and order.price <= self.best_bid_price

        if market_order_buy or market_order_sell:
            mo = MarketOrder(order.volume, order.side)
            return self.handle_market_order(agent_id, mo)
        elif order.price not in self.price_to_orders_ids:
            self.price_to_orders_ids[order.price] = deque([order_id])
            if order.side == Side.BUY:
                insort(self.sorted_bid_prices, order.price)
                self.best_bid_price = self.sorted_bid_prices[-1]
            else:
                insort(self.sorted_ask_prices, order.price)
                self.best_ask_price = self.sorted_ask_prices[0]
            self.mid_price = (self.best_bid_price + self.best_ask_price) / 2
        else:
            self.price_to_orders_ids[order.price].append([order_id])
        self.order_id_to_agent_id[order_id] = agent_id
        self.order_id_to_order[order_id] = order
        return OrderReceipt(True, order_id, "L", order.price, order.volume, order.side)

        def handle_market_order(self, agent_id: str, order: MarketOrder):
            order_id = str(uuid.uuid4())
            volume = order.volume
            if order.side == Side.BUY:
                while volume > 0:
                    price = self.sorted_ask_prices[0]
                    matched_order_id = self.price_to_orders_ids[price][0]
                    matched_order = self.order_id_to_order[matched_order_id]
                    matched_agent_id = self.order_id_to_agent_id[matched_order_id]
                    matched_agent = self.agent_id_to_agent[matched_agent_id]
                    if matched_order.volume > volume:
                    /;#
                    }@+09
                        #TODO communicate to matched agent trading
                        return Order

                    else:
                        pass


            else:
                pass

        def handle_cancel_order(self, order_id):
            # TODO
            raise NotImplementedError
