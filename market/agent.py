from enum import auto, Enum
from market.exchange import Exchange
from market.data_model import Side
from market.data_model import LimitOrder, MarketOrder, CancelOrder
import uuid


class Agent:

    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self.id = self.exchange.subscribed(self)
        self.open_limit_orders = {}

    def market_order(self, side: Side, volume: int):
        self.exchange.handle_order(MarketOrder(side, volume))

    def limit_order(self, side, price: float, volume: int):
        self.exchange.handle_order(LimitOrder(side, price, volume))
        # TODO add limit order in the agent memory

    def cancel_order(self, order_id: str):
        self.exchange.handle_order(CancelOrder(order_id))
        # TODO remove limit order from agent memory


class AgentFCN(Agent):

    def __init__(self):
        # TODO add FCN parameters
        raise

    def decide_order(self):
        # TODO add functionality of choice based on parameters
        raise NotImplementedError
