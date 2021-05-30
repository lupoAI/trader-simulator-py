import matplotlib.pyplot as plt

from market.data_model import Series, MarketSnapshotSeries


class VisualizeSimulation:

    def __init__(self, snapshot: MarketSnapshotSeries, mid_price: Series):
        self.snapshot = snapshot
        self.mid_price = mid_price

    def plot_order_book(self):
        plt.imshow(self.snapshot.price_to_volume_df)
        plt.show()


if __name__ == "__main__":
    from market.simulator import RandomSimulator
    from market.exchange import Exchange

    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(1000, 10, 10000, 20)
    visualizer = VisualizeSimulation(random_simulator.market_snapshots,
                                     random_simulator.mid_price_series)
    visualizer.plot_order_book()
