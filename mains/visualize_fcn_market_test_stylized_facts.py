from analysis.market_analyzer import MarketVisualizer
from market.simulator import SimulatorFCN
from market.exchange import Exchange
from market.data_model import Side


def visualize_fundamental_agents_test_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True


def visualize_momentum_agents_test_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 0)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True


def visualize_noise_agents_test_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 1)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True


def visualize_fundamental_agents_with_trend_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0, 0.0001)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True


def visualize_momentum_agents_with_trend_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 0, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True


def visualize_noise_agents_with_trend_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.4, 1, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True


def visualize_normal_simulation_stylized_facts():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0.002, 1, 0.2, 1, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    simulation_price = simulator_fcn.last_mid_price_series
    simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

    simulated_market_visualizer.visualize_market(30, '../results/random_simulation_1m_rets_market.jpg')
    assert True
