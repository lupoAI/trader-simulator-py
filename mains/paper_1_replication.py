import numpy as np

from analysis.market_analyzer import MarketVisualizer
from analysis.simulation_visualizer import VisualizeSimulation
from market.exchange import Exchange
from market.simulator import SimulatorPaper1
import pandas as pd
import os

if not os.path.exists("../results/paper_1_replication/"):
    os.mkdir("../results/paper_1_replication/")

exchange = Exchange()
simulator_paper_1 = SimulatorPaper1(exchange, 10000, 10000, 10000000, 1000)
simulator_paper_1.run(25200 * 20, 600, 20, 0.005, 1, 2000)
visualizer = VisualizeSimulation(simulator_paper_1)
visualizer.plot_order_book()
visualizer.plot_save_and_show("../results/paper_1_replication/paper1_lbo_heatmap.jpg")


headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

price_df = pd.Series(simulator_paper_1.mid_price_series.price, index=simulator_paper_1.mid_price_series.time_step)
long_price_df = pd.Series(index=np.arange(0, simulator_paper_1.mid_price_series.time_step[-1] + 1), dtype=np.float64)
long_price_df.loc[price_df.index] = price_df.values
long_price_df = long_price_df.ffill().dropna()
minute_price_df = long_price_df.loc[::60]

simulated_market_visualizer = MarketVisualizer(minute_price_df.values, is_simulated=True)

real_market_visualizer.compare_market(1, simulated_market_visualizer,
                                      '../results/paper_1_replication/facts_comparison.jpg')


