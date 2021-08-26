from analysis.market_analyzer import MarketVisualizer
from market.simulator import SimulatorFCN
from market.exchange import Exchange
from market.data_model import Side
import matplotlib.pyplot as plt


def visualize_fundamental_agents_test_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series

    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)
    #
    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts'
    #                                               + f'/fcn_simulation_fundamental_agents_test_{rets_int}_rets.jpg'))
    return simulation_price.price, simulator_fcn.fund_price_series


def visualize_momentum_agents_test_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 0)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)
    #
    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts/'
    #                                               + f'fcn_simulation_momentum_agents_test_{rets_int}_rets.jpg'))
    return simulation_price.price, simulator_fcn.fund_price_series


def visualize_noise_agents_test_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 1)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts/'
    #                                               + f'fcn_simulation_noise_agents_test_{rets_int}_rets.jpg'))
    return simulation_price.price, simulator_fcn.fund_price_series


def visualize_fundamental_agents_with_trend_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)
    #
    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts/'
    #                                               + 'fcn_simulation_fundamental_agents_with_'
    #                                               + f'trend_test_{rets_int}_rets.jpg'))

    return simulation_price.price, simulator_fcn.fund_price_series


def visualize_momentum_agents_with_trend_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 0, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)
    #
    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts/'
    #                                               + 'fcn_simulation_momentum_agents_with_'
    #                                               + f'trend_test_{rets_int}_rets.jpg'))
    return simulation_price.price, simulator_fcn.fund_price_series


def visualize_noise_agents_with_trend_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.4, 1, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)
    #
    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts/'
    #                                               + f'fcn_simulation_noise_agents_with_trend_test_{rets_int}_rets.jpg'))
    return simulation_price.price, simulator_fcn.fund_price_series


def visualize_normal_simulation_stylized_facts(rets_int):
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0.002, 1, 0.2, 1, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    # simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)
    #
    # simulated_market_visualizer.visualize_market(rets_int,
    #                                              ('../results/visualize_fcn_market_test_stylized_facts/'
    #                                               + f'fcn_simulation_normal_{rets_int}_rets.jpg'))
    return simulation_price.price, simulator_fcn.fund_price_series


if __name__ == '__main__':
    import os

    if not os.path.exists("../results/visualize_fcn_market_test_stylized_facts/"):
        os.mkdir("../results/visualize_fcn_market_test_stylized_facts/")

    # For some reason we can only run one at a time otherwise are appending prices

    fund_agents = visualize_fundamental_agents_test_stylized_facts(30)
    mom_agents = visualize_momentum_agents_test_stylized_facts(30)
    noise_agents = visualize_noise_agents_test_stylized_facts(30)
    fund_agents_trend = visualize_fundamental_agents_with_trend_stylized_facts(30)
    mom_agents_trend = visualize_momentum_agents_with_trend_stylized_facts(30)
    noise_agents_trend = visualize_noise_agents_with_trend_stylized_facts(30)
    _ = visualize_normal_simulation_stylized_facts(30)

    ax1 = plt.subplot2grid(shape=(4, 6), loc=(0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    ax4 = plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
    ax6 = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)

    ax1.plot(fund_agents[0], label='mid-price', color='green')
    ax1.plot(fund_agents[1], label='fund-price', color='red')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(mom_agents[0], label='mid-price', color='green')
    ax2.plot(mom_agents[1], label='fund-price', color='red')
    ax2.grid(True)
    ax2.legend()

    ax3.plot(noise_agents[0], label='mid-price', color='green')
    ax3.plot(noise_agents[1], label='fund-price', color='red')
    ax3.grid(True)
    ax3.legend()

    ax4.plot(fund_agents_trend[0], label='mid-price', color='green')
    ax4.plot(fund_agents_trend[1], label='fund-price', color='red')
    ax4.grid(True)
    ax4.legend()

    ax5.plot(mom_agents_trend[0], label='mid-price', color='green')
    ax5.plot(mom_agents_trend[1], label='fund-price', color='red')
    ax5.grid(True)
    ax5.legend()

    ax6.plot(noise_agents_trend[0], label='mid-price', color='green')
    ax6.plot(noise_agents_trend[1], label='fund-price', color='red')
    ax6.grid(True)
    ax6.legend()

    plt.show()




