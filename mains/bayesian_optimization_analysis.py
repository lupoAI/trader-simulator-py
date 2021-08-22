import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.optimizer import spherical_to_cartesian
from analysis.loss_function import aggregate_losses
from market.exchange import Exchange
from market.simulator import SimulatorFCNGamma
from analysis.market_analyzer import MarketVisualizer
from analysis.loss_function import LossFunction
from analysis.simulation_visualizer import VisualizeSimulationFCN
from skopt.plots import plot_convergence


def values_to_proportion(x, y, z):
    x_prop = x / (x + y + z)
    y_prop = y / (x + y + z)
    z_prop = z / (x + y + z)
    return x_prop, y_prop, z_prop


def log_to_linear(x, y, z):
    return np.exp(x), np.exp(y), np.exp(z)


def visualize_parameters_performance(n_iters, scale_fund, scale_chart, scale_noise, target_market_visualizer,
                                     save_dir=None):
    gamma_traders_percentage = 0#0.1

    n_agents = 1000
    initial_fund_price = 5000
    fund_price_vol = 0.002
    fund_price_trend = 0

    n_steps = 16000
    trades_per_step = 4
    snapshot_interval = 30
    cancel_order_interval = 40

    rets_int = [1, 5, 15, 30, "1d"]

    simulator_parameters = {"scale_fund": scale_fund,
                            "scale_chart": scale_chart,
                            "scale_noise": scale_noise,
                            "n_agents": n_agents,
                            "initial_fund_price": initial_fund_price,
                            "fund_price_vol": fund_price_vol,
                            "fund_price_trend": fund_price_trend,
                            "gamma_traders_percentage": gamma_traders_percentage}

    run_parameters = {"n_steps": n_steps,
                      "average_trades_per_step": trades_per_step,
                      "snapshot_interval": snapshot_interval,
                      "cancel_order_interval": cancel_order_interval}

    total_loss = {}
    for i in range(n_iters):

        random_seed_simulation = 105  # np.random.randint(0, 1e6)  # 42
        random_seed_run = 300  # np.random.randint(0, 1e6)  # 42
        print(f"Random seed for simulation is {random_seed_simulation}")
        print(f"Random seed for run is {random_seed_run}")

        simulator_parameters['random_seed'] = random_seed_simulation
        run_parameters['random_seed'] = random_seed_run

        exchange = Exchange()
        simulator_fcn = SimulatorFCNGamma(exchange, **simulator_parameters)
        simulator_fcn.run(**run_parameters)

        visualizer = VisualizeSimulationFCN(simulator_fcn)
        visualizer.plot_order_book()
        visualizer.plot_save_and_show(save_dir + f"lbo_iter_{i}.jpg")

        simulation_price = simulator_fcn.last_mid_price_series
        simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

        losses = []
        for ret in rets_int:
            if save_dir is not None:
                target_market_visualizer.compare_market(ret, simulated_market_visualizer,
                                                        save_dir + f'market_comp_iter_{i}_rets_{ret}.jpg')
            if ret == '1d':
                loss = LossFunction(target_market_visualizer.market_analyzer.get_daily_market_metrics(),
                                    simulated_market_visualizer.market_analyzer.get_daily_market_metrics())
            else:
                loss = LossFunction(target_market_visualizer.market_analyzer.get_market_metrics(ret),
                                    simulated_market_visualizer.market_analyzer.get_market_metrics(ret))
            loss.compute_loss()
            losses += [loss]

        target_close_correlation = target_market_visualizer.market_analyzer.get_close_auto_correlation()
        simulated_close_correlation = simulated_market_visualizer.market_analyzer.get_close_auto_correlation()
        correlation_loss = mean_absolute_error(target_close_correlation, simulated_close_correlation)

        plt.plot(target_close_correlation, label='target correlation')
        plt.plot(simulated_close_correlation, label='simulated correlation')
        plt.legend()
        plt.title('Close Correlation Loss')
        plt.xlabel('Correlation')
        if save_dir is not None:
            plt.savefig(save_dir + f'correlation_loss_iter_{i}.jpg')
        plt.show()

        total_loss[i] = aggregate_losses(losses)['total_loss'] * 0.5 + correlation_loss * 0.5

    print('total_loss:')
    print(total_loss)

    with open(save_dir + 'total_loss.pickle', 'wb') as loss_dir:
        pickle.dump(total_loss, loss_dir)

    return total_loss


def mean_absolute_error(target, simulated):
    return (np.abs(target - simulated)).mean()


if __name__ == "__main__":
    import os

    # TODO Add volatility multiplier

    test_number = 11
    test_path = f"../results/bayesian_optimization_training/test_{test_number}/"
    test_path_analysis = f"../results/bayesian_optimization_analysis/test_{test_number}/"
    n_sims = 2

    if not os.path.exists("../results/bayesian_optimization_analysis"):
        os.mkdir("../results/bayesian_optimization_analysis")

    if not os.path.exists(test_path_analysis):
        os.mkdir(test_path_analysis)

    with open(test_path + 'test_results.pickle', 'rb') as test_results:
        res = pickle.load(test_results)

    plot_convergence(res)

    fund, chart, noise = spherical_to_cartesian(1.5, *res.x)
    fund, chart, noise = log_to_linear(fund, chart, noise)
    fund, chart, noise = values_to_proportion(fund, chart, noise)

    print(f'The best percentages of fundamental, momentum and noise are: {fund}, {chart}, {noise}')

    headers = ['Open', 'High', 'Low', 'Close']
    data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
    data = data.drop(columns=[5])
    data.columns = headers
    real_market_visualizer = MarketVisualizer(data)

    visualize_parameters_performance(n_sims, fund, chart, noise, real_market_visualizer, test_path_analysis)
