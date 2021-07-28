import pandas as pd
from temp.fcn_simulator_update import SimulatorFCN
from market.exchange import Exchange
from analysis.market_analyzer import MarketVisualizer
import os

if not os.path.exists("../results/_test/compare_markets/"):
    os.mkdir("../results/_test/compare_markets/")

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

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

real_market_visualizer.compare_market(1, simulated_market_visualizer,
                                      '../results/_test/compare_markets/comparison_real_fake_1_rets.jpg')
real_market_visualizer.compare_market(5, simulated_market_visualizer,
                                      '../results/_test/compare_markets/comparison_real_fake_5_rets.jpg')
real_market_visualizer.compare_market(15, simulated_market_visualizer,
                                      '../results/_test/compare_markets/comparison_real_fake_15_rets.jpg')
real_market_visualizer.compare_market(30, simulated_market_visualizer,
                                      '../results/_test/compare_markets/comparison_real_fake_30_rets.jpg')
real_market_visualizer.compare_market("1d", simulated_market_visualizer,
                                      '../results/_test/compare_markets/comparison_real_fake_1d_rets.jpg')


# TODO find close stylized fact
# TODO aggregate GMB benchmark