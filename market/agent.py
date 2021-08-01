from numpy import log, exp, random

from market.data_model import LimitOrder, MarketOrder, CancelOrder
from market.data_model import OrderReceipt
from market.data_model import Side
from market.exchange import Exchange

# TODO figure out actual value
DAY_TRADING_MINUTES = 4000  # 23400


class Agent:

    def __init__(self, exchange: Exchange):
        self.exchange = exchange
        self.id = self.exchange.subscribed(self)
        self.open_limit_orders = {}
        self.cash = 0
        self.stock = 0
        self.value_portfolio = 0

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
        if order_receipt.side == Side.BUY:
            self.cash -= order_receipt.average_price * order_receipt.volume
            self.stock += order_receipt.volume
        elif order_receipt.side == Side.SELL:
            self.cash += order_receipt.average_price * order_receipt.volume
            self.stock -= order_receipt.volume
        else:
            raise ValueError("Order Side should be either BUY or SELL")

    def handle_cancel_order_receipt(self, order_id: str):
        del self.open_limit_orders[order_id]

    def handle_limit_execution(self, order_id: str, volume: int):
        order = self.open_limit_orders[order_id]
        price = order.price
        order_volume = order.volume
        if volume > order_volume:
            raise ValueError("Requested volume is greater than offered volume")
        if order.side == Side.BUY:
            self.cash -= price * volume
            self.stock += volume
        else:
            self.cash += price * volume
            self.stock -= volume
        if order.volume == volume:
            del self.open_limit_orders[order_id]
            self.exchange.closed_limit_order_routine(order_id)
        else:
            order.trade_volume(volume)

    def compute_value_of_portfolio(self):
        value = self.cash
        if self.exchange.last_valid_mid_price is not None:
            value += self.stock * self.exchange.last_valid_mid_price
        self.value_portfolio = value

    def endow(self, cash, stock):
        self.cash += cash
        self.stock += stock


class AgentFCN(Agent):

    def __init__(self, exchange: Exchange, f_param: float, c_param: float, n_param: float,
                 time_window: int, order_margin: float):
        super().__init__(exchange)
        self.f_param = f_param
        self.c_param = c_param
        self.n_param = n_param
        self.time_window = time_window
        self.order_margin = order_margin
        self.current_fundamental_price = None
        self.starting_price = None
        self.current_mid_price = None
        self.current_best_ask = None
        self.current_best_bid = None
        self.previous_mid_price = None
        self.noise_sample = None

    def submit_parameters(self):
        return {'f_param': self.f_param,
                'c_param': self.c_param,
                'n_param': self.n_param,
                'time_window': self.time_window,
                'order_margin': self.order_margin}

    def submit_attributes(self):
        self.compute_value_of_portfolio()
        return {'cash': self.cash,
                'stock': self.stock,
                'value_portfolio': self.value_portfolio}

    def get_data(self, current_fundamental_price: float, starting_price: int, previous_mid_price: float,
                 noise_sample: float):
        self.current_fundamental_price = current_fundamental_price
        self.starting_price = starting_price
        self.current_mid_price = self.exchange.last_valid_mid_price
        if self.current_mid_price is None:
            self.current_mid_price = self.starting_price
        self.current_best_bid = self.exchange.best_bid_price
        if self.current_mid_price is None:
            self.current_mid_price = self.starting_price
        self.current_best_ask = self.exchange.best_ask_price
        if self.current_best_ask is None:
            self.current_best_ask = self.starting_price
        self.previous_mid_price = previous_mid_price
        if self.previous_mid_price is None:
            self.previous_mid_price = self.starting_price
        self.noise_sample = noise_sample

    def submit_time_window(self):
        return self.time_window

    def decide_order(self, volume: int, buy_outcome: float = random.uniform()):
        # TODO figure out what is T (DAY_TRADING_MINUTES)
        # TODO add tests for this function
        f = log(self.current_fundamental_price / self.current_mid_price) / DAY_TRADING_MINUTES
        c = log(self.current_mid_price / self.previous_mid_price) / DAY_TRADING_MINUTES
        n = self.noise_sample
        r = (self.f_param * f + self.c_param * c + self.n_param * n) / (self.f_param + self.c_param + self.n_param)
        future_price = self.current_mid_price * exp(r * self.time_window)

        buy_probability = 1 / (1 + exp(-r / 0.002))
        limit_mult = 1

        if buy_outcome <= buy_probability and future_price > self.current_mid_price:
            price_to_buy = int(future_price * (1 - self.order_margin) + 0.5)
            volume_to_buy = volume if price_to_buy >= self.current_mid_price else int(volume * limit_mult)
            return self.limit_order(Side.BUY, price_to_buy, volume_to_buy, True)
        elif buy_outcome <= buy_probability and future_price <= self.current_mid_price:
            price_to_buy = int(self.current_mid_price * (1 - self.order_margin) + 0.5)
            volume_to_buy = volume if price_to_buy >= self.current_mid_price else int(volume * limit_mult)
            return self.limit_order(Side.BUY, price_to_buy, volume_to_buy, True)
        elif buy_outcome > buy_probability and future_price > self.current_mid_price:
            price_to_sell = int(self.current_mid_price * (1 + self.order_margin) + 0.5)
            volume_to_sell = volume if price_to_sell <= self.current_mid_price else int(volume * limit_mult)
            return self.limit_order(Side.SELL, price_to_sell, volume_to_sell, True)
        else:
            price_to_sell = int(future_price * (1 + self.order_margin) + 0.5)
            volume_to_sell = volume if price_to_sell <= self.current_mid_price else int(volume * limit_mult)
            return self.limit_order(Side.SELL, price_to_sell, volume_to_sell, True)


class AgentGamma(Agent):

    def __init__(self, exchange: Exchange, leverage):
        super().__init__(exchange)
        self.starting_price = None
        self.current_mid_price = None
        self.current_best_ask = None
        self.current_best_bid = None
        self.previous_mid_price = None
        self.leverage = leverage

    def submit_attributes(self):
        self.compute_value_of_portfolio()
        return {'cash': self.cash,
                'stock': self.stock,
                'value_portfolio': self.value_portfolio}

    def get_data(self, starting_price: int, previous_mid_price: float):
        self.starting_price = starting_price
        self.current_mid_price = self.exchange.last_valid_mid_price
        if self.current_mid_price is None:
            self.current_mid_price = self.starting_price
        self.current_best_bid = self.exchange.best_bid_price
        if self.current_mid_price is None:
            self.current_mid_price = self.starting_price
        self.current_best_ask = self.exchange.best_ask_price
        if self.current_best_ask is None:
            self.current_best_ask = self.starting_price
        self.previous_mid_price = previous_mid_price
        if self.previous_mid_price is None:
            self.previous_mid_price = self.starting_price

    def decide_order(self):
        daily_return = self.current_mid_price / self.previous_mid_price - 1
        volume_traded = int(daily_return * 100 * self.leverage + 0.5)
        if volume_traded == 0:
            return self.market_order(Side.BUY, 0, True)
        elif volume_traded > 0:
            return self.market_order(Side.BUY, volume_traded, True)
        else:
            return self.market_order(Side.SELL, -volume_traded, True)
