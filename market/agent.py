from typing import Union

from market.data_model import LimitOrder, MarketOrder, CancelOrder
from market.data_model import OrderReceipt
from market.data_model import Side
from market.exchange import Exchange


class Agent:

    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self.id = self.exchange.subscribed(self)
        self.open_limit_orders = {}
        self.cash = 0
        self.stock = 0

    def market_order(self, side: Side, volume: int, return_receipt: bool = False):
        mo = MarketOrder(volume, side)
        order_receipt = self.exchange.handle_order(self.id, mo)
        if order_receipt.outcome:
            self.handle_market_order_receipt(order_receipt)
        if return_receipt:
            return order_receipt

    def limit_order(self, side, price: int, volume: int, return_receipt: bool = False):
        lo = LimitOrder(price, volume, side)
        order_receipt = self.exchange.handle_order(self.id, lo)
        if order_receipt.outcome:
            self.handle_limit_order_receipt(order_receipt, lo)
        if return_receipt:
            return order_receipt

    def cancel_order(self, order_id: str, return_receipt: bool = False):
        co = CancelOrder(order_id)
        order_receipt = self.exchange.handle_order(self.id, co)
        if order_receipt.outcome:
            self.handle_cancel_order_receipt(order_id)
        if return_receipt:
            return order_receipt

    def handle_limit_order_receipt(self, order_receipt: OrderReceipt, order: LimitOrder):
        self.open_limit_orders[order_receipt.order_id] = order

    def handle_market_order_receipt(self, order_receipt: OrderReceipt):
        if order_receipt == Side.BUY:
            self.cash -= order_receipt.average_price * order_receipt.volume
            self.stock += order_receipt.volume
        else:
            self.cash += order_receipt.average_price * order_receipt.volume
            self.stock -= order_receipt.volume

    def handle_cancel_order_receipt(self, order_id: str):
        del self.open_limit_orders[order_id]

    def handle_limit_execution(self, order_id: str, average_price: Union[float, int], volume: int):
        # TODO
        raise NotImplementedError

    def get_order_ids(self):
        return list(self.open_limit_orders.keys())


class AgentFCN(Agent):

    def __init__(self):
        # super().__init__()
        # TODO add FCN parameters
        raise NotImplementedError

    def decide_order(self):
        # TODO add functionality of choice based on parameters
        raise NotImplementedError
