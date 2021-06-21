from analysis.visualizer import VisualizeSimulation
from market.exchange import Exchange
from market.simulator import SimulatorPaper1

exchange = Exchange()
simulator_paper_1 = SimulatorPaper1(exchange, 10000, 10000, 100000, 1000)

simulator_paper_1.run(25200*50, 600, 20, 0.005, 1, 600)
visualizer = VisualizeSimulation(simulator_paper_1)
visualizer.plot_order_book()
visualizer.plot_save_and_show("../results/paper1_lbo_heatmap.jpg")