from analysis.market_analyzer import SimulatedMarketAnalyzer, MarketVisualizer
from market.simulator import RandomSimulator
from market.exchange import Exchange


def test_simulated_market_analyzer_works():
    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(5000, 20, 500, 5, 20)
    prices = random_simulator.last_mid_price_series.price
    simulation_analyzer = SimulatedMarketAnalyzer(prices)
    _ = simulation_analyzer.get_market_metrics(1)
    _ = simulation_analyzer.get_market_metrics(15)
    assert True


def test_deleting_market_analyzer_works():
    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(500, 20, 500, 5, 20)
    prices = random_simulator.last_mid_price_series.price
    market_visualizer = MarketVisualizer(prices, is_simulated=True)
    market_visualizer.visualize_market(15)
    del exchange
    del random_simulator
    del market_visualizer
    exchange = Exchange()
    random_simulator = RandomSimulator(exchange, 100)
    random_simulator.run(500, 20, 500, 5, 20)
    prices = random_simulator.last_mid_price_series.price
    market_visualizer = MarketVisualizer(prices, is_simulated=True)
    market_visualizer.visualize_market(15)
    assert len(prices) == 500
