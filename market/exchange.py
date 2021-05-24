import uuid

from market.data_model import Order
from market.data_model import LimitOrder, MarketOrder, CancelOrder
#from market.agent import Agent


class Exchange:

    def __init__(self):
        self.price_to_orders = {}
        self.orders_id_to_price = {}
        self.orders_id_to_agent = {}
        self.subscribed_agents = {}
        self.best_bid = 0
        self.best_ask = 0
        # TODO
        #raise NotImplementedError

    def subscribed(self, agent_self):
        agent_id = str(uuid.uuid4())
        self.subscribed_agents[agent_id] = agent_self
        return agent_id

    def handle_order(self, order: Order):
        order_id = str(uuid.uuid4())
        if order.type == "L":
            raise NotImplementedError
        elif order.type == "M":
            raise NotImplementedError
        elif order.type == "C":
            raise NotImplementedError
        else:
            raise NotImplementedError
