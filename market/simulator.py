from dataclasses import dataclass, field

from numpy import exp, array
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
        adjustment = (sides - 1.5) * 0.1
        sign = rand_state.randint(0, 2, size=size_matrix) * 2 - 1
        volume = 1
        offset = rand_state.uniform(0, 0.03, size=size_matrix)
        orders_to_cancel = {}
        for i in range(n_steps):
            if i % snapshot_interval == 0 and i > 0:
                self.save_market_snapshot(i)

            orders_to_cancel[i] = []
            for j in range(trades_per_step):
                if self.exchange.last_valid_mid_price is None:
                    price = int((starting_price + adjustment[i, j])
                                * (1 + sign[i, j] * offset[i, j]))
                else:
                    price = int((self.exchange.last_valid_mid_price + adjustment[i, j])
                                * (1 + sign[i, j] * offset[i, j]))
                order_receipt = self.agents[agents[i, j]].limit_order(Side(sides[i, j]), price, volume, True)
                orders_to_cancel[i] += [(agents[i, j], order_receipt.order_id)]

            self.mid_price_series.add(i, self.exchange.mid_price)

            if i >= cancel_order_interval:
                for agent_id, order_id in orders_to_cancel[i - cancel_order_interval]:
                    self.agents[agent_id].cancel_order(order_id)

        self.market_snapshots.format_price_to_volume()


class SimulatorFCN(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int, initial_fund_price: int, fund_price_vol: float,
                 random_seed: int = 42):
        super().__init__(exchange, n_agents)
        self.fund_price_vol = fund_price_vol
        self.initial_fund_price = initial_fund_price
        rand_state = RandomState(random_seed)
        self.scale_fund = rand_state.normal(0.3, 0.03)
        self.scale_chart = rand_state.normal(0.3, 0.03)
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
                    previous_price = self.mid_price_series[-time_window]
                self.agents[agents[i, j]].get_data(self.fund_price_series[i], self.initial_fund_price, previous_price,
                                                   agents_noise[i, j])
                order_receipt = self.agents[agents[i, j]].decide_order(agents_volume[i, j])
                orders_to_cancel[i] += [(agents[i, j], order_receipt.order_id)]

            self.mid_price_series.add(i, self.exchange.mid_price)

            if i >= cancel_order_interval:
                for agent_id, order_id in orders_to_cancel[i - cancel_order_interval]:
                    self.agents[agent_id].cancel_order(order_id)
        self.market_snapshots.format_price_to_volume()
