from market.simulator import SimulatorFCNGamma
from market.exchange import Exchange
from analysis.simulation_visualizer import VisualizeSimulationFCN
import matplotlib.pyplot as plt
from numpy import round, array, sqrt
import os
import pickle

if not os.path.exists('../results/visualize_fcn_gamma/'):
    os.makedirs('../results/visualize_fcn_gamma/')

gamma_traders_percentage = 0#0.1

scale_fund = 0.2
scale_chart = 0.1
scale_noise = 0.7
n_agents = 1000
initial_fund_price = 5000
fund_price_vol = 0.002
fund_price_trend = 0
random_seed_simulation = 42

n_steps = 10000
trades_per_step = 5
snapshot_interval = 5
cancel_order_interval = 20
random_seed_run = 42

simulator_parameters = {"scale_fund": scale_fund,
                        "scale_chart": scale_chart,
                        "scale_noise": scale_noise,
                        "n_agents": n_agents,
                        "initial_fund_price": initial_fund_price,
                        "fund_price_vol": fund_price_vol,
                        "fund_price_trend": fund_price_trend,
                        'gamma_traders_percentage': gamma_traders_percentage,
                        "random_seed": random_seed_simulation}

run_parameters = {"n_steps": n_steps,
                  "average_trades_per_step": trades_per_step,
                  "snapshot_interval": snapshot_interval,
                  "cancel_order_interval": cancel_order_interval,
                  "random_seed": random_seed_run}

exchange = Exchange()
simulator_fcn = SimulatorFCNGamma(exchange, **simulator_parameters)
simulator_fcn.run(**run_parameters)
visualizer = VisualizeSimulationFCN(simulator_fcn)
visualizer.plot_order_book()
visualizer.plot_save_and_show("../results/visualize_fcn_gamma/fcn_lbo_heatmap_1000.jpg")
plt.plot(simulator_fcn.minutes_of_trading, simulator_fcn.ewma_square_returns)
plt.title(f"EWMA of square returns: alpha {round(simulator_fcn.ewma_alpha, 3)}")
plt.savefig("../results/visualize_fcn_gamma/EWMA_returns.jpg")
plt.show()
volatility_multiplier = sqrt(simulator_fcn.ewma_square_returns) / 0.004 # array(simulator_fcn.ewma_square_returns) / 0.004 ** 2
volatility_multiplier[volatility_multiplier < 0.6] = 0.6
volatility_multiplier[volatility_multiplier > 3] = 3
plt.plot(simulator_fcn.minutes_of_trading, volatility_multiplier)
plt.title(f"Volatility Multiplier: alpha {round(simulator_fcn.ewma_alpha, 3)}")
plt.savefig("../results/visualize_fcn_gamma/volatility_multiplier.jpg")
plt.show()

with open('../results/visualize_fcn_gamma/minutes_of_trading.pickle', 'wb') as mins:
    pickle.dump(simulator_fcn.minutes_of_trading, mins)

with open('../results/visualize_fcn_gamma/square_returns.pickle', 'wb') as sq_ret:
    pickle.dump(simulator_fcn.square_returns, sq_ret)

with open('../results/visualize_fcn_gamma/ewma_square_returns.pickle', 'wb') as ewma_sq_ret:
    pickle.dump(simulator_fcn.ewma_square_returns, ewma_sq_ret)

with open('../results/visualize_fcn_gamma/price_series.pickle', 'wb') as ser:
    pickle.dump(simulator_fcn.last_mid_price_series, ser)
