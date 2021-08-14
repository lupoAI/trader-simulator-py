import os
import pandas as pd
import numpy as np
from numpy.random import RandomState
from analysis.market_analyzer import MarketVisualizer
from analysis.loss_function import compute_total_loss
from market.exchange import Exchange
from market.simulator import RandomSimulator
from analysis.simulation_visualizer import VisualizeSimulation

if not os.path.exists("../results/compare_losses/"):
    os.mkdir("../results/compare_losses/")

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

# GBM


list_rets = [1, 5, 15, 30, 390]

n_series = 1
n_steps = 200000
T = 10000
vol = 0.2
trend = - 0.5 * vol ** 2
random_seed = 100
starting_price = 10000
save = True
show_samples = False

random_state = RandomState(random_seed)
gauss = random_state.normal(size=(n_steps, n_series))
gbm = np.exp(trend / T + np.sqrt(1 / T) * vol * gauss)
gbm = starting_price * gbm.cumprod(axis=0)

simulated_market_visualizer = MarketVisualizer(gbm[:, 0], is_simulated=True)
real_market_visualizer.compare_market("1d", simulated_market_visualizer,
                                      '../results/compare_losses/GBM_stats.jpg')

total_loss = compute_total_loss(gbm[:, 0], "GBM")
print(f"Total Loss wrt SPX: {total_loss}")

# Random Simulator

exchange = Exchange()
random_simulator = RandomSimulator(exchange, 100)
random_simulator.run(200000, 20, 500, 5, 20)

# visualizer = VisualizeSimulation(random_simulator)
# visualizer.plot_order_book()
# visualizer.plot_save_and_show("../results/compare_losses/random_lbo.jpg")

simulated_market_visualizer = MarketVisualizer(random_simulator.last_mid_price_series.price, is_simulated=True)

real_market_visualizer.compare_market('1d', simulated_market_visualizer,
                                      '../results/compare_losses/RandomSimulator_stats.jpg')

total_loss = compute_total_loss(random_simulator.last_mid_price_series.price, "RandomSimulator")
print(f"Total Loss wrt SPX: {total_loss}")

# Exponential Simulator

# FCN Simulator

# FCN Gamma Simulator
