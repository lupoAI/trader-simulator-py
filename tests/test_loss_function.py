from analysis.market_analyzer import SimulatedMarketAnalyzer
from market.simulator import RandomSimulator
from market.exchange import Exchange
from analysis.loss_function import LossFunction


def test_simulated_market_analyzer_works():
    exchange_1 = Exchange()
    random_simulator_1 = RandomSimulator(exchange_1, 100)
    random_simulator_1.run(5000, 20, 500, 5, 20)
    prices_1 = random_simulator_1.last_mid_price_series.price
    simulation_analyzer_1 = SimulatedMarketAnalyzer(prices_1)
    style_1 = simulation_analyzer_1.get_market_metrics(15)

    exchange_2 = Exchange()
    random_simulator_2 = RandomSimulator(exchange_2, 100)
    random_simulator_2.run(4000, 20, 500, 5, 20, 10)
    prices_2 = random_simulator_2.last_mid_price_series.price
    simulation_analyzer_2 = SimulatedMarketAnalyzer(prices_2)
    style_2 = simulation_analyzer_2.get_market_metrics(15)

    loss = LossFunction(style_1, style_2)
    loss.compute_loss()

    assertions = []
    assertions += [loss.auto_correlation_loss is not None]
    assertions += [loss.volatility_clustering_loss is not None]
    assertions += [loss.leverage_effect_loss is not None]
    assertions += [loss.distribution_loss is not None]
    assertions += [loss.total_loss is not None]
    assert assertions == [True] * 5
