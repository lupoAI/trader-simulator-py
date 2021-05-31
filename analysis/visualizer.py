from typing import Union
import matplotlib.pyplot as plt

from market.data_model import Series, MarketSnapshotSeries


class VisualizeSimulation:

    def __init__(self, snapshot: MarketSnapshotSeries, mid_price: Series):
        self.snapshot = snapshot
        self.mid_price = mid_price

    def plot_order_book(self, save_name: Union[None, str] = None):
        scatter_data = self.snapshot.price_to_volume_df.stack().dropna()
        x = scatter_data.index.get_level_values(0)
        y = scatter_data.index.get_level_values(1)
        z = scatter_data.values
        plt.scatter(x, y, c=z)
        plt.plot(self.mid_price.time_step, self.mid_price.price)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()


if __name__ == "__main__":
    from market.simulator import RandomSimulator
    from market.exchange import Exchange

    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(1000, 10, 10000, 20, 60)
    visualizer = VisualizeSimulation(random_simulator.market_snapshots,
                                     random_simulator.mid_price_series)
    visualizer.plot_order_book("../results/random_lbo_heatmap.jpg")
