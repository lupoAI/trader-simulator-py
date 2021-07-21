# Uncomment elements in tests to see the results otherwise just look at

# from analysis.simulation_visualizer import VisualizeSimulationFCN
from market.simulator import SimulatorFCN
from market.exchange import Exchange
from market.data_model import Side


def test_fundamental_agents_work():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_only_fundamental_check.jpg")
    assert True


def test_momentum_agents_work():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 0)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_momentum_check.jpg")
    assert True


def test_noise_agents_work():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 1)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_noise_check.jpg")
    assert True


def test_fundamental_agents_with_trend():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0, 0.0001)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_fundamental_with_trend.jpg")
    assert True


def test_momentum_agents_with_trend():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.2, 0, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_momentum_with_trend.jpg")
    assert True


def test_noise_agents_with_trend():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0.4, 1, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_noise_with_trend.jpg")
    assert True


def test_normal_simulation():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0.002, 1, 0.2, 1, 0.00005)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(5000, 10, 5, 20)
    # visualizer = VisualizeSimulationFCN(simulator_fcn)
    # visualizer.plot_order_book()
    # visualizer.plot_save_and_show("../results/test_fcn_simulator/fcn_lbo_normal_simulation.jpg")
    assert True
