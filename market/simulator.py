import numpy as np

from market.agent import Agent
from market.data_model import Side
from market.exchange import Exchange


class Simulator:

    def __init__(self, exchange: Exchange, n_agents: int):
        self.exchange = exchange
        self.n_agents = n_agents


class RandomSimulator(Simulator):

    def __init__(self, exchange: Exchange, n_agents: int):
        super().__init__(exchange, n_agents)
        self.agents = [Agent(self.exchange) for _ in range(self.n_agents)]

    def run(self, n_steps: int, trades_per_step: int, starting_price: int):
        size_matrix = (n_steps, trades_per_step)
        agents = np.random.randint(0, self.n_agents, size=size_matrix)
        sides = np.random.randint(1, 3, size=size_matrix)
        adjustment = (sides - 1.5) * 0.2
        sign = np.random.randint(0, 2, size=size_matrix) * 2 - 1
        volume = 1
        offset = np.random.uniform(0, 0.1, size=size_matrix)
        mid_price = []
        for i in range(n_steps):
            for j in range(trades_per_step):
                if self.exchange.last_valid_mid_price is None:
                    price = int((starting_price + adjustment[i, j])
                                * (1 + sign[i, j] * offset[i, j]))
                else:
                    price = int((self.exchange.last_valid_mid_price + adjustment[i, j])
                                * (1 + sign[i, j] * offset[i, j]))
                self.agents[agents[i, j]].limit_order(Side(sides[i, j]), price, volume)
            mid_price += [self.exchange.mid_price]
        return mid_price
