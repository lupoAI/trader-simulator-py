from analysis.market_analyzer import MarketVisualizer
from market.simulator import SimulatorFCN
from analysis.loss_function import LossFunction
from market.exchange import Exchange
import pandas as pd
import os
from itertools import product
from copy import deepcopy
from tqdm import tqdm


if not os.path.exists("../results/parameters_grid_search/"):
    os.mkdir("../results/parameters_grid_search/")


headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)


scale_fund = list(range(5, 11))
scale_chart = list(range(5, 11))
scale_noise = list(range(5, 11))


n_agents = 100
initial_fund_price = 5000
fund_price_vol = 0.002
fund_price_trend = 0
random_seed_simulation = 42
n_steps = 3000
trades_per_step = 2
snapshot_interval = 5
cancel_order_interval = 20
random_seed_run = 42
rets_int = 30

simulator_parameters = {"n_agents": n_agents,
                        "initial_fund_price": initial_fund_price,
                        "fund_price_vol": fund_price_vol,
                        "fund_price_trend": fund_price_trend,
                        "random_seed": random_seed_simulation}

run_parameters = {"n_steps": n_steps,
                  "trades_per_step": trades_per_step,
                  "snapshot_interval": snapshot_interval,
                  "cancel_order_interval": cancel_order_interval,
                  "random_seed": random_seed_run}

for (fund, chart, noise) in tqdm(product(scale_fund, scale_chart, scale_noise)):

    simulator_parameters['scale_fund'] = fund
    simulator_parameters['scale_chart'] = chart
    simulator_parameters['scale_noise'] = noise

    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, **simulator_parameters)
    simulator_fcn.run(**run_parameters)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    # Check that things are going well in the loop
    # simulated_market_visualizer.visualize_market(rets_int)

    loss = LossFunction(real_market_visualizer.market_analyzer.get_market_metrics(rets_int),
                        simulated_market_visualizer.market_analyzer.get_market_metrics(rets_int))
    loss.compute_loss()

    loss_dict = {'auto_correlation_loss': loss.auto_correlation_loss,
                 'volatility_clustering_loss': loss.volatility_clustering_loss,
                 'leverage_effect_loss': loss.leverage_effect_loss,
                 'distribution_loss': loss.distribution_loss,
                 'total_loss': loss.total_loss}

    simulator_parameters_copy = deepcopy(simulator_parameters)
    run_parameters_copy = deepcopy(run_parameters)

    simulator_parameters_copy['random_seed_simulator'] = random_seed_simulation
    del simulator_parameters_copy['random_seed']
    run_parameters_copy['random_seed_run'] = random_seed_run
    del run_parameters_copy['random_seed']

    test_results = {**simulator_parameters_copy, **run_parameters_copy, **loss_dict}

    results_df = pd.DataFrame(test_results, index=[0])

    log_file = '../results/parameters_grid_search/grid_search_tests.csv'

    if os.path.exists(log_file):
        results_df.to_csv(log_file, index=False, mode='a', header=False)
    else:
        results_df.to_csv(log_file, index=False)

    del exchange
    del simulator_fcn
    del simulated_market_visualizer
