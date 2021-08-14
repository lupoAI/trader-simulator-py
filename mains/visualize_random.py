from analysis.market_analyzer import MarketVisualizer
from analysis.simulation_visualizer import VisualizeSimulation
from market.simulator import RandomSimulator
from market.exchange import Exchange
from analysis.loss_function import compute_total_loss
import pandas as pd
import os

if not os.path.exists("../results/visualize_random/"):
    os.mkdir("../results/visualize_random/")

exchange = Exchange()
random_simulator = RandomSimulator(exchange, 100)
random_simulator.run(1000, 10, 500, 1, 20)
visualizer = VisualizeSimulation(random_simulator)
visualizer.plot_order_book()
visualizer.plot_save_and_show("../results/visualize_random/random_lbo_heatmap.jpg")
del exchange
del random_simulator
del visualizer


exchange = Exchange()
random_simulator = RandomSimulator(exchange, 100)
random_simulator.run(5000, 20, 500, 5, 20)
visualizer = VisualizeSimulation(random_simulator)
visualizer.plot_order_book()
visualizer.plot_save_and_show("../results/visualize_random/random_lbo_heatmap_1000.jpg")


headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

simulated_market_visualizer = MarketVisualizer(random_simulator.last_mid_price_series.price, is_simulated=True)


real_market_visualizer.compare_market(1, simulated_market_visualizer,
                                      '../results/visualize_random/facts_comparison.jpg')

total_loss = compute_total_loss(random_simulator.last_mid_price_series.price)
print(f"Total Loss wrt SPX: {total_loss}")