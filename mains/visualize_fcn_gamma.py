from market.simulator import SimulatorFCNGamma
from market.exchange import Exchange
from analysis.simulation_visualizer import VisualizeSimulationFCN
from analysis.market_analyzer import MarketVisualizer
from analysis.loss_function import compute_total_loss
from analysis.optimizer import spherical_to_cartesian
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle


def values_to_proportion(x, y, z):
    x_prop = x / (x + y + z)
    y_prop = y / (x + y + z)
    z_prop = z / (x + y + z)
    return x_prop, y_prop, z_prop


def log_to_linear(x, y, z):
    return np.exp(x), np.exp(y), np.exp(z)


if not os.path.exists('../results/visualize_fcn_gamma/'):
    os.makedirs('../results/visualize_fcn_gamma/')

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

test_number = 23
test_path = f"../results/bayesian_optimization_training/test_{test_number}/"

with open(test_path + 'test_results.pickle', 'rb') as test_results:
    res = pickle.load(test_results)

plot_convergence(res)

fund, chart, noise = spherical_to_cartesian(1.5, *res.x[:2])
fund, chart, noise = log_to_linear(fund, chart, noise)
fund, chart, noise = values_to_proportion(fund, chart, noise)

scale_fund = fund
scale_chart = chart
scale_noise = noise
n_agents = 1000
initial_fund_price = 5000
fund_price_vol = 0.002
fund_price_trend = 0
random_seed_simulation = 42

n_steps = 200000
gamma_traders_percentage = res.x[2]
trades_per_step = int(res.x[7])
snapshot_interval = 1000
cancel_order_interval = int(res.x[3])
random_seed_run = 42
lam = res.x[8]
order_margin = res.x[4]
min_lookback = int(res.x[5])
lookback_range = int(res.x[6])

simulator_parameters = {"scale_fund": scale_fund,
                        "scale_chart": scale_chart,
                        "scale_noise": scale_noise,
                        "n_agents": n_agents,
                        "initial_fund_price": initial_fund_price,
                        "fund_price_vol": fund_price_vol,
                        "fund_price_trend": fund_price_trend,
                        'gamma_traders_percentage': gamma_traders_percentage,
                        "random_seed": random_seed_simulation,
                        "lam": lam,
                        "order_margin": order_margin,
                        "min_lookback": min_lookback,
                        "lookback_range": lookback_range}

run_parameters = {"n_steps": n_steps,
                  "average_trades_per_step": trades_per_step,
                  "snapshot_interval": snapshot_interval,
                  "cancel_order_interval": cancel_order_interval,
                  "random_seed": random_seed_run}

exchange = Exchange()
simulator_fcn = SimulatorFCNGamma(exchange, **simulator_parameters)
simulator_fcn.run(**run_parameters)
# visualizer = VisualizeSimulationFCN(simulator_fcn)
# visualizer.plot_order_book()
# visualizer.plot_save_and_show("../results/visualize_fcn_gamma/fcn_lbo_heatmap_1000.jpg")
# plt.plot(simulator_fcn.minutes_of_trading, simulator_fcn.ewma_std_list)
# plt.title(f"EWMA of square returns: lam {np.round(simulator_fcn.lam, 3)}")
# plt.savefig("../results/visualize_fcn_gamma/EWMA_returns.jpg")
# plt.show()
# ewma_std_list = np.array(simulator_fcn.ewma_std_list)
# std_list = np.array(simulator_fcn.std_list)
# volatility_multiplier = np.exp(2 * (ewma_std_list / std_list - 1))
# volatility_multiplier[volatility_multiplier < 0.5] = 0.5
# volatility_multiplier[volatility_multiplier > 5] = 3
# plt.plot(simulator_fcn.minutes_of_trading, volatility_multiplier)
# plt.title(f"Volatility Multiplier: lam {np.round(simulator_fcn.lam, 3)}")
# plt.savefig("../results/visualize_fcn_gamma/volatility_multiplier.jpg")
# plt.show()

simulated_market_visualizer = MarketVisualizer(simulator_fcn.last_mid_price_series.price, is_simulated=True)

real_market_visualizer.compare_market(1, simulated_market_visualizer,
                                      '../results/visualize_fcn_gamma/fcn_gamma_1m.jpg')
real_market_visualizer.compare_market(5, simulated_market_visualizer,
                                      '../results/visualize_fcn_gamma/fcn_gamma_5m.jpg')
real_market_visualizer.compare_market(15, simulated_market_visualizer,
                                      '../results/visualize_fcn_gamma/fcn_gamma_15m.jpg')
real_market_visualizer.compare_market(30, simulated_market_visualizer,
                                      '../results/visualize_fcn_gamma/fcn_gamma_30m.jpg')
real_market_visualizer.compare_market('1d', simulated_market_visualizer,
                                      '../results/visualize_fcn_gamma/fcn_gamma_1d.jpg')

real_market_visualizer.compare_close_auto_correlation(simulated_market_visualizer,
                                                      "../results/visualize_fcn_gamma/fcn_gamma_close.jpg")

total_loss = compute_total_loss(simulator_fcn.last_mid_price_series.price, "FCNSimulatorGamma")
print(f"Total Loss wrt SPX: {total_loss}")

with open('../results/visualize_fcn_gamma/minutes_of_trading.pickle', 'wb') as mins:
    pickle.dump(simulator_fcn.minutes_of_trading, mins)

with open('../results/visualize_fcn_gamma/square_returns.pickle', 'wb') as sq_ret:
    pickle.dump(simulator_fcn.square_returns, sq_ret)

with open('../results/visualize_fcn_gamma/ewma_square_returns.pickle', 'wb') as ewma_sq_ret:
    pickle.dump(simulator_fcn.ewma_std_list, ewma_sq_ret)

with open('../results/visualize_fcn_gamma/price_series.pickle', 'wb') as ser:
    pickle.dump(simulator_fcn.last_mid_price_series, ser)
