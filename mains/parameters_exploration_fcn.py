from analysis.market_analyzer import MarketVisualizer
from market.simulator import SimulatorFCN
from analysis.loss_function import LossFunction
from analysis.simulation_visualizer import VisualizeSimulationFCN
from market.exchange import Exchange
import pandas as pd
from glob import glob
import os

if not os.path.exists("../results/parameters_exploration_fcn/"):
    os.mkdir("../results/parameters_exploration_fcn/")

tests = glob("../results/parameters_exploration_fcn/*/")

new_test = len(tests) + 1

# Uncomment next line to alter some tests
# new_test = 1

test_path = f"../results/parameters_exploration_fcn/test_{new_test}/"

if not os.path.exists(test_path):
    os.mkdir(test_path)

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

n_agents = 100
initial_fund_price = 5000
fund_price_vol = 0.002
scale_fund = 2
scale_chart = 0.4
scale_noise = 1
fund_price_trend = 0
random_seed_simulation = 42
n_steps = 10000
trades_per_step = 2
snapshot_interval = 5
cancel_order_interval = 20
random_seed_run = 42
rets_int = 30

simulator_parameters = {"n_agents": n_agents,
                        "initial_fund_price": initial_fund_price,
                        "fund_price_vol": fund_price_vol,
                        "scale_fund": scale_fund,
                        "scale_chart": scale_chart,
                        "scale_noise": scale_noise,
                        "fund_price_trend": fund_price_trend,
                        "random_seed": random_seed_simulation}

run_parameters = {"n_steps": n_steps,
                  "trades_per_step": trades_per_step,
                  "snapshot_interval": snapshot_interval,
                  "cancel_order_interval": cancel_order_interval,
                  "random_seed": random_seed_run}

exchange = Exchange()
simulator_fcn = SimulatorFCN(exchange, **simulator_parameters)
simulator_fcn.run(**run_parameters)
simulation_price = simulator_fcn.last_mid_price_series
simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

# Visualize LBO
visualizer = VisualizeSimulationFCN(simulator_fcn)
visualizer.plot_order_book()
visualizer.plot_save_and_show(test_path + 'price_with_lbo.jpg')

# Visualize Stylized Facts
simulated_market_visualizer.visualize_market(rets_int, test_path + f'/stylized_facts_{rets_int}_rets.jpg')

# Visualize comparison to Target Data
real_market_visualizer.compare_market(rets_int, simulated_market_visualizer,
                                      test_path + f'comparison_real_fake_{rets_int}_rets.jpg')

loss = LossFunction(real_market_visualizer.market_analyzer.get_market_metrics(rets_int),
                    simulated_market_visualizer.market_analyzer.get_market_metrics(rets_int))
loss.compute_loss()

loss_dict = {'auto_correlation_loss': loss.auto_correlation_loss,
             'volatility_clustering_loss': loss.volatility_clustering_loss,
             'leverage_effect_loss': loss.leverage_effect_loss,
             'distribution_loss': loss.distribution_loss,
             'total_loss': loss.total_loss}

simulator_parameters['random_seed_simulator'] = random_seed_simulation
del simulator_parameters['random_seed']
run_parameters['random_seed_run'] = random_seed_run
del run_parameters['random_seed']

test_results = {**simulator_parameters, **run_parameters, **loss_dict}
test_results['test_path'] = test_path

results_df = pd.DataFrame(test_results, index=[new_test])

log_file = '../results/parameters_exploration_fcn/tests_log.csv'

if os.path.exists(log_file):
    results_df.to_csv(log_file, index=False, mode='a', header=False)
else:
    results_df.to_csv(log_file, index=False)
