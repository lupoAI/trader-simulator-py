from typing import List
from typing import Union

import matplotlib.pyplot as plt
from numpy import meshgrid, unique, max, full, nan

from market.data_model import Series, MarketSnapshotSeries


class VisualizeSimulation:

    def __init__(self, snapshot: MarketSnapshotSeries, mid_price: Series):
        self.snapshot = snapshot
        self.mid_price = mid_price

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
        _ = ax.pcolormesh(xm, ym, zm, cmap='viridis', vmin=0, vmax=max(zm), shading='nearest')
        if add_mid:
            self.add_mid_price()
        if add_bid_ask:
            self.add_best_bid()
            self.add_best_ask()

    @staticmethod
    def plot_save_and_show(save_name: Union[None, str] = None):
        plt.legend()
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()

    def add_mid_price(self):
        plt.plot(self.mid_price.time_step, self.mid_price.price, color='green', label='mid price')

    def add_best_bid(self):
        plt.plot(self.snapshot.time_step, self.snapshot.best_bid_price, color='orange', label='best bid')

    def add_best_ask(self):
        plt.plot(self.snapshot.time_step, self.snapshot.best_ask_price, color='blue', label='best ask')


class VisualizeSimulationFCN(VisualizeSimulation):

    def __init__(self, snapshot: MarketSnapshotSeries, mid_price: Series, fundamental_price: List):
        super().__init__(snapshot, mid_price)
        self.fundamental_price = fundamental_price

    def plot_order_book(self, add_mid: bool = True, add_bid_ask: bool = False, add_fund: bool = True):
        super().plot_order_book(add_mid, add_bid_ask)
        self.add_fund()

    def add_fund(self):
        ind = list(range(len(self.fundamental_price)))
        plt.plot(ind, self.fundamental_price, color='red', label='fundamental price')


if __name__ == "__main__":
    from market.simulator import RandomSimulator
    from market.exchange import Exchange

    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(1000, 10, 1000, 5, 20)
    visualizer = VisualizeSimulation(random_simulator.market_snapshots,
                                     random_simulator.mid_price_series)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/random_lbo_heatmap.jpg")

if __name__ == "__main__":
    from market.simulator import SimulatorFCN
    from market.exchange import Exchange

    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0.00001)
    simulator_fcn.run(1000, 10, 5, 20)
    visualizer = VisualizeSimulationFCN(simulator_fcn.market_snapshots,
                                        simulator_fcn.mid_price_series,
                                        simulator_fcn.fund_price_series)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/fcn_lbo_heatmap.jpg")
