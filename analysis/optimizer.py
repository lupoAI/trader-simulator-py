import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from market.exchange import Exchange
from market.simulator import SimulatorFCNGamma
from analysis.market_analyzer import MarketVisualizer
from analysis.loss_function import LossFunction, aggregate_losses
from skopt import gp_minimize
from skopt.plots import plot_convergence
from multiprocessing import Pool

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)


def spherical_to_cartesian(rho, theta, phi):
    x = rho * np.sin(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.sin(theta)
    z = rho * np.cos(phi)
    return x, y, z


def simulate_market(params):
    print(params)
    rho = 1.5
    theta = params[0]
    phi = params[1]

    x, y, z = spherical_to_cartesian(rho, theta, phi)

    scale_fund = np.exp(x)
    scale_chart = np.exp(y)
    scale_noise = np.exp(z)

    gamma_traders_percentage = 0

    n_agents = 1000
    initial_fund_price = 5000
    fund_price_vol = 0.002
    fund_price_trend = 0
    random_seed_simulation = np.random.randint(0, 1e6)  # 42

    n_steps = 20000
    trades_per_step = 2
    snapshot_interval = 5
    cancel_order_interval = 20
    random_seed_run = np.random.randint(0, 1e6)  # 42

    rets_int = [1, 5, 15, 30, "1d"]

    simulator_parameters = {"scale_fund": scale_fund,
                            "scale_chart": scale_chart,
                            "scale_noise": scale_noise,
                            "n_agents": n_agents,
                            "initial_fund_price": initial_fund_price,
                            "fund_price_vol": fund_price_vol,
                            "fund_price_trend": fund_price_trend,
                            "gamma_traders_percentage": gamma_traders_percentage,
                            "random_seed": random_seed_simulation}

    run_parameters = {"n_steps": n_steps,
                      "average_trades_per_step": trades_per_step,
                      "snapshot_interval": snapshot_interval,
                      "cancel_order_interval": cancel_order_interval,
                      "random_seed": random_seed_run}

    exchange = Exchange()
    simulator_fcn = SimulatorFCNGamma(exchange, **simulator_parameters)
    simulator_fcn.run(**run_parameters)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    # Check that things are going well in the loop
    # real_market_visualizer.compare_market(rets_int, simulated_market_visualizer)

    global real_market_visualizer

    losses = []
    for ret in rets_int:
        # Check that things are going well in the loop
        # target_market_visualizer.compare_market(ret, simulated_market_visualizer,
        #                                         save_dir + f'market_comp_iter_{i}_rets_{ret}.jpg')

        if ret == '1d':
            loss = LossFunction(real_market_visualizer.market_analyzer.get_daily_market_metrics(),
                                simulated_market_visualizer.market_analyzer.get_daily_market_metrics())
        else:
            loss = LossFunction(real_market_visualizer.market_analyzer.get_market_metrics(ret),
                                simulated_market_visualizer.market_analyzer.get_market_metrics(ret))
        loss.compute_loss()
        losses += [loss]

    correlation_loss = mean_absolute_error(real_market_visualizer.market_analyzer.get_close_auto_correlation(),
                                           simulated_market_visualizer.market_analyzer.get_close_auto_correlation())

    total_loss = aggregate_losses(losses)

    final_loss = correlation_loss + total_loss["total_loss"] * 5

    log_loss = np.log(final_loss)

    return log_loss


def mean_absolute_error(target, simulated):
    return (np.abs(target - simulated)).mean()


def simulate_market_multiprocessing(params):
    cores_number = 5
    p = Pool(processes=cores_number)
    log_loss_list = p.map(simulate_market, [params] * cores_number)
    log_loss_list = np.array(log_loss_list)
    loss_list = np.exp(log_loss_list)
    mean_loss = loss_list.mean()
    log_loss = np.log(mean_loss)

    return log_loss


def use_bayesian_optimization(bounds, acq_func, n_calls, n_random_starts, noise, random_state, save_name=None):
    res = gp_minimize(simulate_market_multiprocessing,
                      bounds,
                      acq_func=acq_func,
                      n_calls=n_calls,
                      n_random_starts=n_random_starts,
                      noise=noise,
                      random_state=random_state)
    plot_convergence(res)
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()
    return res
