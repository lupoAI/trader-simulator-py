import pandas as pd
from market.simulator import SimulatorFCN
from market.exchange import Exchange
from analysis.market_analyzer import MarketVisualizer
import os
import matplotlib.pyplot as plt

if not os.path.exists("../results/visualize_fcn_correlation_loss/"):
    os.mkdir("../results/visualize_fcn_correlation_loss/")

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

real_market_close_auto_correlation = real_market_visualizer.market_analyzer.get_close_auto_correlation()

exchange = Exchange()

n_agents = 100
initial_fund_price = 5000
fund_price_vol = 0.002
scale_fund = 0.2
scale_chart = 0.1
scale_noise = 0.7


fcn_simulator = SimulatorFCN(exchange, n_agents, initial_fund_price, fund_price_vol,
                             scale_fund, scale_chart, scale_noise)

fcn_simulator.run(10000, 5, 100, 40, 20)

simulation_price = fcn_simulator.last_mid_price_series
simulated_market_visualizer = MarketVisualizer(pd.Series(simulation_price.price), is_simulated=True)

simulated_market_close_auto_correlation = simulated_market_visualizer.market_analyzer.get_close_auto_correlation()

plt.plot(list(range(1, 6)), real_market_close_auto_correlation, label="real_market")
plt.plot(list(range(1, 6)), simulated_market_close_auto_correlation, label="simulated_market")
plt.title("Correlation Close Intra-Day Performance")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("../results/visualize_fcn_correlation_loss/close_stylized_fact.jpg")
plt.show()


