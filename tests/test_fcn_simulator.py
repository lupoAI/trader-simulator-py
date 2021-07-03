from analysis.simulation_visualizer import VisualizeSimulationFCN
from market.simulator import SimulatorFCN
from market.exchange import Exchange
from market.data_model import Side


def test_fundamental_agents_work():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 0, 0.3)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(1000, 10, 5, 20)
    visualizer = VisualizeSimulationFCN(simulator_fcn)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/fcn_lbo_only_fundamental_check.jpg")
    assert True


def test_momentum_agents_work():
    exchange = Exchange()
    simulator_fcn = SimulatorFCN(exchange, 100, 500, 0, 1, 1, 0.5)
    simulator_fcn.agents[0].limit_order(Side.BUY, 398, 5)
    simulator_fcn.agents[1].limit_order(Side.SELL, 402, 5)
    simulator_fcn.run(1000, 10, 5, 20)
    visualizer = VisualizeSimulationFCN(simulator_fcn)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/fcn_lbo_momentum_check.jpg")
    assert True