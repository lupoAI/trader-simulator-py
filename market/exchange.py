import uuid
from bisect import insort
from collections import deque
from copy import deepcopy
from typing import Union

from market.data_model import LimitOrder, MarketOrder, CancelOrder
from market.data_model import OrderReceipt
from market.data_model import Side

# TODO add traded volume tracking
# TODO add moving averages
class Exchange:

    def __init__(self):
        self.agent_id_to_agent = {}
        self.order_id_to_order = {}
        self.order_id_to_agent_id = {}
        self.price_to_orders_ids = {}
        self.price_to_volume = {}

        self.sorted_bid_prices = []
        self.sorted_ask_prices = []

        self.best_bid_price = None
        self.best_ask_price = None
        self.best_bid_volume = None
        self.best_ask_volume = None
        self.total_bid_volume = 0
        self.total_ask_volume = 0

        self.mid_price = None
        self.last_valid_mid_price = None

    def subscribed(self, agent_self):
        agent_id = str(uuid.uuid4())
        self.agent_id_to_agent[agent_id] = agent_self
        return agent_id

    def handle_order(self, agent_id: str, order: Union[LimitOrder, MarketOrder, CancelOrder]):
        if order.type == "L":
            return self.handle_limit_order(agent_id, order)
        elif order.type == "M":
            return self.handle_market_order(order)
        elif order.type == "C":
            return self.handle_cancel_order(order)
        else:
            raise ValueError("Order Type not understood")

    def handle_limit_order(self, agent_id: str, order: LimitOrder):
        order_valid = self.check_order_validity(order)
        if not order_valid:
            return OrderReceipt()

        order_id = str(uuid.uuid4())
        market_order_condition = self.get_market_order_conditions(order.side, order.price)

        if market_order_condition:
            mo = MarketOrder(order.volume, order.side)
            return self.handle_market_order(mo)
        elif order.price not in self.price_to_orders_ids:
            self.price_to_orders_ids[order.price] = deque([order_id])
            self.price_to_volume[order.price] = order.volume
            if order.side == Side.BUY:
                insort(self.sorted_bid_prices, order.price)
                self.best_bid_price = self.sorted_bid_prices[-1]
                self.best_bid_volume = self.price_to_volume[self.best_bid_price]
            else:
                insort(self.sorted_ask_prices, order.price)
                self.best_ask_price = self.sorted_ask_prices[0]
                self.best_ask_volume = self.price_to_volume[self.best_ask_price]
            self.set_mid_price()
        else:
            self.price_to_orders_ids[order.price].append(order_id)
            self.price_to_volume[order.price] += order.volume

        if order.side == Side.BUY:
            self.total_bid_volume += order.volume
        else:
            self.total_ask_volume += order.volume

        self.order_id_to_agent_id[order_id] = agent_id
        self.order_id_to_order[order_id] = order
        return OrderReceipt(True, order_id, "L", order.price, order.volume, order.side)

    def handle_market_order(self, order: MarketOrder):
        order_valid = self.check_order_validity(order)
        if not order_valid:
            return OrderReceipt()

        order_id = str(uuid.uuid4())
        volume_left = order.volume
        volume_executed = 0
        total_price = 0
        average_price = 0
        while volume_left > 0:
            if order.side == Side.BUY:
                if not self.sorted_ask_prices:
                    if volume_executed != 0:
                        return OrderReceipt(True, order_id, "M", average_price,
                                            volume_executed, order.side)
                    else:
                        return OrderReceipt()
                price = self.sorted_ask_prices[0]
            else:
                if not self.sorted_bid_prices:
                    if volume_executed != 0:
                        return OrderReceipt(True, order_id, "M", average_price,
                                            volume_executed, order.side)
                    else:
                        return OrderReceipt()
                price = self.sorted_bid_prices[-1]
            matched_order_id = self.price_to_orders_ids[price][0]
            matched_order = self.order_id_to_order[matched_order_id]
            matched_agent_id = self.order_id_to_agent_id[matched_order_id]
            matched_agent = self.agent_id_to_agent[matched_agent_id]
            matched_volume = min(matched_order.volume, volume_left)
            self.price_to_volume[price] -= matched_volume
            matched_agent.handle_limit_execution(matched_order_id, matched_volume)
            volume_left -= matched_volume
            volume_executed += matched_volume
            total_price += matched_volume * price
            average_price = total_price / volume_executed

        if order.side == Side.BUY:
            self.total_ask_volume -= order.volume
            self.best_ask_volume = self.price_to_volume.get(self.best_ask_price, None)
        else:
            self.total_bid_volume -= order.volume
            self.best_bid_volume = self.price_to_volume.get(self.best_bid_price, None)

        return OrderReceipt(True, order_id, "M", average_price, volume_executed, order.side)

    def handle_cancel_order(self, order: CancelOrder):
        order_valid = self.check_order_validity(order)
        if not order_valid:
            return OrderReceipt()

        order_id = order.order_id
        price = self.order_id_to_order[order_id].price
        side = self.order_id_to_order[order_id].side
        volume = self.order_id_to_order[order_id].volume
        orders_at_price = self.price_to_orders_ids[price]
        if len(orders_at_price) == 1:
            del self.price_to_orders_ids[price]
            del self.price_to_volume[price]
            if side == Side.BUY:
                self.sorted_bid_prices.remove(price)
                if self.sorted_bid_prices:
                    self.best_bid_price = self.sorted_bid_prices[-1]
                    self.best_bid_volume = self.price_to_volume[self.best_bid_price]
                    self.total_bid_volume -= volume
                else:
                    self.best_bid_price = None
                    self.best_bid_volume = None
                    self.total_bid_volume = 0
            else:
                self.sorted_ask_prices.remove(price)
                if self.sorted_ask_prices:
                    self.best_ask_price = self.sorted_ask_prices[0]
                    self.best_ask_volume = self.price_to_volume[self.best_ask_price]
                    self.total_ask_volume -= volume
                else:
                    self.best_ask_price = None
                    self.best_ask_volume = None
                    self.total_ask_volume = 0
        else:
            _ = self.price_to_orders_ids[price].remove(order_id)
        self.set_mid_price()
        del self.order_id_to_order[order_id]
        del self.order_id_to_agent_id[order_id]
        return OrderReceipt(True)

    def closed_limit_order_routine(self, order_id):
        price = self.order_id_to_order[order_id].price
        side = self.order_id_to_order[order_id].side
        orders_at_price = self.price_to_orders_ids[price]
        if len(orders_at_price) == 1:
            del self.price_to_orders_ids[price]
            del self.price_to_volume[price]
            if side == Side.BUY:
                del self.sorted_bid_prices[-1]
                if self.sorted_bid_prices:
                    self.best_bid_price = self.sorted_bid_prices[-1]
                    self.best_bid_volume = self.price_to_volume[self.best_bid_price]
                else:
                    self.best_bid_price = None
                    self.best_bid_volume = None
            else:
                del self.sorted_ask_prices[0]
                if self.sorted_ask_prices:
                    self.best_ask_price = self.sorted_ask_prices[0]
                    self.best_ask_volume = self.price_to_volume[self.best_ask_price]
                else:
                    self.best_ask_price = None
                    self.best_ask_volume = None
        else:
            _ = self.price_to_orders_ids[price].popleft()
        self.set_mid_price()
        del self.order_id_to_order[order_id]
        del self.order_id_to_agent_id[order_id]

    def set_mid_price(self):
        if (self.best_bid_price is not None) and (self.best_ask_price is not None):
            self.mid_price = (self.best_bid_price + self.best_ask_price) / 2
        else:
            self.mid_price = None
        if self.mid_price is not None:
            self.last_valid_mid_price = self.mid_price

    def get_market_order_conditions(self, side: Side, price: int):
        if (self.best_bid_price is not None) and (self.best_ask_price is not None):
            market_order_buy = side == Side.BUY and price >= self.best_ask_price
            market_order_sell = side == Side.SELL and price <= self.best_bid_price
            market_order_condition = market_order_buy or market_order_sell
        elif self.best_bid_price is not None:
            market_order_condition = side == Side.SELL and price <= self.best_bid_price
        elif self.best_ask_price is not None:
            market_order_condition = side == Side.BUY and price >= self.best_ask_price
        else:
            market_order_condition = False
        return market_order_condition

    def check_order_validity(self, order: Union[LimitOrder, MarketOrder, CancelOrder]):
        if order.type == "L":
            return order.price > 0 and order.volume > 0
        elif order.type == "M":
            return order.volume > 0
        elif order.type == 'C':
            return order.order_id in self.order_id_to_order
        else:
            raise ValueError("Order Type not understood")

    def return_market_snapshot(self):
        snapshot = {'price_to_volume': self.price_to_volume,
                    'best_bid_price': self.best_bid_price,
                    'best_ask_price': self.best_ask_price,
                    'best_bid_volume': self.best_bid_volume,
                    'best_ask_volume': self.best_ask_volume,
                    'total_bid_volume': self.total_bid_volume,
                    'total_ask_volume': self.total_ask_volume,
                    'mid_price': self.mid_price
                    }
        return deepcopy(snapshot)
