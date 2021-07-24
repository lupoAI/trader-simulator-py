from typing import Union

import matplotlib.pyplot as plt
from numpy import meshgrid, unique, nanmax, full, nan, array

from market.data_model import AgentsInfo
from market.simulator import Simulator, SimulatorFCN


class VisualizeSimulation:

    def __init__(self, simulator: Simulator):
        self.snapshot = simulator.market_snapshots
        self.mid_price = simulator.mid_price_series

    def plot_order_book(self, add_mid: bool = True, add_bid_ask: bool = False):
        lob_data = self.snapshot.price_to_volume_df.stack().dropna()
        x = unique(lob_data.index.get_level_values(0))
        y = unique(lob_data.index.get_level_values(1))
        xm, ym = meshgrid(x, y)
        zm = full(xm.shape, nan)
        for i in range(xm.shape[0]):
            for j in range(ym.shape[1]):
                if (xm[i, j], ym[i, j]) in lob_data.index:
                    zm[i, j] = lob_data.loc[(xm[i, j], ym[i, j])]
        fig, ax = plt.subplots()
        _ = ax.pcolormesh(xm, ym, zm, cmap='viridis', vmin=0, vmax=nanmax(zm), shading='nearest')
        if add_mid:
            self.add_mid_price()
        if add_bid_ask:
            self.add_best_bid()
            self.add_best_ask()

    @staticmethod
    def plot_save_and_show(save_name: Union[None, str] = None, has_legend: bool = True):
        if has_legend:
            plt.legend()
        if save_name is not None:
            plt.savefig(save_name)
        plt.show(block=False)

    def add_mid_price(self):
        plt.plot(self.mid_price.time_step, self.mid_price.price, color='green', label='mid price')

    def add_best_bid(self):
        plt.plot(self.snapshot.time_step, self.snapshot.best_bid_price, color='orange', label='best bid')

    def add_best_ask(self):
        plt.plot(self.snapshot.time_step, self.snapshot.best_ask_price, color='blue', label='best ask')


class VisualizeSimulationFCN(VisualizeSimulation):

    def __init__(self, simulator: SimulatorFCN):
        super().__init__(simulator)
        self.fundamental_price = simulator.fund_price_series
        self.agents = simulator.agents
        self.agents_info = AgentsInfo()

    def compute_agents_info(self):
        for agent in self.agents:
            self.agents_info.add_parameters(agent.submit_parameters())
            self.agents_info.add_attributes(agent.submit_attributes())

    def plot_agents_attributes_vs_parameters(self, attribute: str):
        ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
        ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
        ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
        f_param = array(self.agents_info.f_param)
        c_param = array(self.agents_info.c_param)
        n_param = array(self.agents_info.n_param)
        f_w = f_param / (f_param + c_param + n_param)
        c_w = c_param / (f_param + c_param + n_param)
        n_w = n_param / (f_param + c_param + n_param)
        ax1.scatter(f_w, self.agents_info.__getattribute__(attribute), color='orange', marker='+')
        ax1.set_xlabel('f_param')
        ax2.scatter(c_w, self.agents_info.__getattribute__(attribute), color='green', marker='h')
        ax2.set_xlabel('c_param')
        ax3.scatter(n_w, self.agents_info.__getattribute__(attribute), color='red', marker='^')
        ax3.set_xlabel('n_param')
        ax4.scatter(self.agents_info.time_window, self.agents_info.__getattribute__(attribute), color='purple',
                    marker='x')
        ax4.set_xlabel('time_window')
        ax5.scatter(self.agents_info.order_margin, self.agents_info.__getattribute__(attribute), color='blue',
                    marker='d')
        ax5.set_xlabel('order_margin')
        plt.suptitle(attribute)

    def plot_order_book(self, add_mid: bool = True, add_bid_ask: bool = False, add_fund: bool = True):
        super().plot_order_book(add_mid, add_bid_ask)
        self.add_fund()

    def add_fund(self):
        ind = list(range(len(self.fundamental_price)))
        plt.plot(ind, self.fundamental_price, color='red', label='fundamental price')
