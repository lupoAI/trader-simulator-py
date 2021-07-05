from dataclasses import dataclass, field
from math import gamma

import numpy as np
from numpy import exp, array, log, power, cumsum
from numpy.random import RandomState

from market.agent import Agent, AgentFCN
from market.data_model import MarketSnapshotSeries, MarketSnapshot
from market.data_model import Side, Series
from market.exchange import Exchange


@dataclass
class Simulator:
    exchange: Exchange
    n_agents: int
    market_snapshots: MarketSnapshotSeries = field(default=MarketSnapshotSeries(), init=False)
    mid_price_series: Series = field(default=Series(), init=False)
    last_mid_price_series: Series = field(default=Series(), init=False)

    def save_market_snapshot(self, time_step: int):
        market_snapshot = self.exchange.return_market_snapshot()
        self.market_snapshots.add(MarketSnapshot(time_step=time_step, **market_snapshot))

    def clear_cache(self):
        self.market_snapshots = MarketSnapshotSeries()
        self.mid_price_series = Series()


class RandomSimulator(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int):
        super().__init__(exchange, n_agents)
        self.agents = [Agent(self.exchange) for _ in range(self.n_agents)]

    def run(self, n_steps: int, trades_per_step: int, starting_price: int, snapshot_interval: int,
            cancel_order_interval: int, random_seed: int = 42):
        self.clear_cache()
        rand_state = RandomState(random_seed)
        size_matrix = (n_steps, trades_per_step)
        agents = rand_state.randint(0, self.n_agents, size=size_matrix)
        sides = rand_state.randint(1, 3, size=size_matrix)
        volume = 1
        vol = 0.02
        mu = 0
        vol_adjustment = -vol ** 2 / 2
        offset = rand_state.normal(0, vol, size=size_matrix)
        multiplier = np.exp(offset + vol_adjustment + mu)
        orders_to_cancel = {}
        for i in range(n_steps):
            if i % snapshot_interval == 0 and i > 0:
                self.save_market_snapshot(i)

            orders_to_cancel[i] = []
            for j in range(trades_per_step):
                if self.exchange.last_valid_mid_price is None:
                    price = int(starting_price * multiplier[i, j] + 0.5)
                else:
                    price = int(self.exchange.last_valid_mid_price * multiplier[i, j] + 0.5)
                order_receipt = self.agents[agents[i, j]].limit_order(Side(sides[i, j]), price, volume, True)
                orders_to_cancel[i] += [(agents[i, j], order_receipt.order_id)]

            self.mid_price_series.add(i, self.exchange.mid_price)
            self.last_mid_price_series.add(i, self.exchange.last_valid_mid_price)

            if i >= cancel_order_interval:
                for agent_id, order_id in orders_to_cancel[i - cancel_order_interval]:
                    self.agents[agent_id].cancel_order(order_id)

        self.market_snapshots.format_price_to_volume()


class SimulatorFCN(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, initial_fund_price: int, fund_price_vol: float,
                 scale_fund=None, scale_chart=None, scale_noise=None, random_seed: int = 42):
        super().__init__(exchange, n_agents)
        self.fund_price_vol = fund_price_vol
        self.initial_fund_price = initial_fund_price
        rand_state = RandomState(random_seed)
        if scale_fund is not None:
            self.scale_fund = scale_fund
        else:
            self.scale_fund = rand_state.normal(0.3, 0.03)
        if scale_chart is not None:
            self.scale_chart = scale_chart
        else:
            self.scale_chart = rand_state.normal(0.3, 0.03)
        if scale_noise is not None:
            self.scale_noise = scale_noise
        else:
            self.scale_noise = rand_state.normal(0.3, 0.03)
        self.agents_fcn = rand_state.exponential(size=(n_agents, 3))
        self.agents_fcn *= array([[self.scale_fund, self.scale_chart, self.scale_noise]])
        self.agents_time_window = rand_state.randint(500, 1001, size=(n_agents,))
        self.agents_order_margin = rand_state.uniform(0, 0.05, size=(n_agents,))
        self.agents = [AgentFCN(self.exchange,
                                *self.agents_fcn[i],
                                self.agents_time_window[i],
                                self.agents_order_margin[i]) for i in range(self.n_agents)]
        self.fund_price_series = array([])

    def create_fundamental_price_series(self, n_steps, random_seed: int):
        rand_state = RandomState(random_seed)
        evolution = rand_state.normal(size=n_steps)
        percentage_change = exp(evolution * self.fund_price_vol)
        time_series = percentage_change.cumprod() * self.initial_fund_price
        self.fund_price_series = time_series

    def clear_cache(self):
        super().clear_cache()
        self.fund_price_series = array([])

    def run(self, n_steps: int, trades_per_step: int, snapshot_interval: int,
            cancel_order_interval: int, random_seed: int = 42):
        self.clear_cache()
        rand_state = RandomState(random_seed)
        self.create_fundamental_price_series(n_steps, rand_state.randint(1000))
        size_matrix = (n_steps, trades_per_step)
        agents = rand_state.randint(0, self.n_agents, size=size_matrix)
        agents_noise = rand_state.normal(0, 0.0001, size=size_matrix)
        agents_volume = rand_state.randint(1, 5, size=size_matrix)
        orders_to_cancel = {}
        for i in range(n_steps):
            if i % snapshot_interval == 0 and i > 0:
                self.save_market_snapshot(i)

            orders_to_cancel[i] = []
            for j in range(trades_per_step):
                time_window = self.agents[agents[i, j]].submit_time_window()
                if time_window > i - 1:
                    previous_price = None
                else:
                    previous_price = self.last_mid_price_series[-time_window]
                self.agents[agents[i, j]].get_data(self.fund_price_series[i], self.initial_fund_price, previous_price,
                                                   agents_noise[i, j])
                order_receipt = self.agents[agents[i, j]].decide_order(agents_volume[i, j])
                orders_to_cancel[i] += [(agents[i, j], order_receipt.order_id)]

            self.mid_price_series.add(i, self.exchange.mid_price)
            self.last_mid_price_series.add(i, self.exchange.last_valid_mid_price)

            if i >= cancel_order_interval:
                for agent_id, order_id in orders_to_cancel[i - cancel_order_interval]:
                    self.agents[agent_id].cancel_order(order_id)
        self.market_snapshots.format_price_to_volume()


class SimulatorPaper1(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, starting_price: int, starting_cash: int, starting_stock: int):
        super().__init__(exchange, n_agents)
        self.starting_price = starting_price
        self.starting_cash = starting_cash
        self.starting_stock = starting_stock
        self.agents = [Agent(self.exchange) for _ in range(self.n_agents)]
        for agent in self.agents:
            agent.endow(starting_cash, starting_stock)

    def run(self, time_simulation: int, snapshot_interval: int, mean_wait_time: int, margin_vol: float, beta: float,
            lifespan_order: int, random_seed: int = 42):
        super().clear_cache()
        rand_state = RandomState(random_seed)
        number_draws = 2 * int(time_simulation / mean_wait_time)
        rand_uniform = rand_state.uniform(size=number_draws)
        eta = mean_wait_time / gamma(1 / beta + 1)
        weibull_waiting_times = eta * power(-log(rand_uniform), 1 / beta)
        weibull_order_times = cumsum(weibull_waiting_times)
        if weibull_order_times[-1] < time_simulation:
            raise ValueError(f"Not enough time order simulation: {weibull_order_times[-1]}")
        weibull_order_times = (weibull_order_times[weibull_order_times <= time_simulation] + 0.5).astype(int)
        number_trades = len(weibull_order_times)
        chosen_agents = rand_state.randint(0, len(self.agents), size=number_trades)
        chosen_side = rand_state.randint(1, 3, size=number_trades)
        chosen_margin = rand_state.normal(1, margin_vol, size=number_trades)
        chosen_quantity = rand_state.uniform(size=number_trades)
        orders = [(x, "O", i) for i, x in enumerate(weibull_order_times)]
        cancellations = [(x + lifespan_order, "C", i) for i, x in enumerate(weibull_order_times)]
        actions = sorted(orders + cancellations)

        order_id_to_exchange_order_id = {}
        previous_time = 0
        for action in actions:

            time = action[0]
            type_action = action[1]
            idx = action[2]

            if time // snapshot_interval > previous_time // snapshot_interval:
                self.save_market_snapshot(time)
            previous_time = time

            agent_id = chosen_agents[idx]
            side = chosen_side[idx]
            margin = chosen_margin[idx]
            quantity = chosen_quantity[idx]
            if type_action == "C":
                order_id = order_id_to_exchange_order_id[idx]
                self.agents[agent_id].cancel_order(order_id)
            elif type_action == "O":

                if side == 1:
                    best_bid = self.exchange.best_bid_price
                    if self.exchange.last_valid_mid_price is not None:
                        best_bid = best_bid if best_bid is not None else int(
                            self.exchange.last_valid_mid_price * 0.995 + 0.5)
                    best_bid = best_bid if best_bid is not None else int(self.starting_price * 0.995 + 0.5)
                    order_volume = (quantity * self.agents[agent_id].cash / best_bid + 0.5)
                    order_receipt = self.agents[agent_id].limit_order(Side.BUY, int(0.9975 * best_bid * margin + 0.5),
                                                                      order_volume, True)
                else:
                    best_ask = self.exchange.best_ask_price
                    if self.exchange.last_valid_mid_price is not None:
                        best_ask = best_ask if best_ask is not None else int(
                            self.exchange.last_valid_mid_price * 1.005 + 0.5)
                    best_ask = best_ask if best_ask is not None else int(self.starting_price * 1.005 + 0.5)
                    order_volume = int(quantity * self.agents[agent_id].stock + 0.5)
                    order_receipt = self.agents[agent_id].limit_order(Side.SELL, int(1.0025 * best_ask * margin + 0.5),
                                                                      order_volume, True)

                order_id_to_exchange_order_id[idx] = order_receipt.order_id
                self.mid_price_series.add(time, self.exchange.mid_price)
            else:
                raise ValueError("Wrong action type")
        self.market_snapshots.format_price_to_volume()
