from analysis.simulation_visualizer import VisualizeSimulation
from market.exchange import Exchange
from market.simulator import SimulatorPaper1

if __name__ == "__main__":
    exchange = Exchange()
    simulator_paper_1 = SimulatorPaper1(exchange, 10000, 10000, 10000000, 1000)
    simulator_paper_1.run(25200 * 10, 600, 20, 0.005, 1, 2000)
    visualizer = VisualizeSimulation(simulator_paper_1)
    visualizer.plot_order_book()
    visualizer.plot_save_and_show("../results/paper1_lbo_heatmap.jpg")
