from dataclasses import dataclass, field

from numpy.random import RandomState

from market.agent import Agent
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
            if (i + 1) % snapshot_interval == 0:
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
