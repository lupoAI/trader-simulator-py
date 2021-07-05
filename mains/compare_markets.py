import pandas as pd
from market.simulator import RandomSimulator
from market.exchange import Exchange
from analysis.market_analyzer import MarketVisualizer

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_visualizer = MarketVisualizer(data)

exchange = Exchange()
random_simulator = RandomSimulator(exchange, 100)
random_simulator.run(5000, 20, 500, 5, 20)
simulation_price = random_simulator.last_mid_price_series
simulated_market_visualizer = MarketVisualizer(pd.Series(simulation_price.price), is_simulated=True)

real_market_visualizer.compare_market(1, simulated_market_visualizer, '../results/comparison_real_fake_1_rets.jpg')
real_market_visualizer.compare_market(5, simulated_market_visualizer, '../results/comparison_real_fake_5_rets.jpg')
real_market_visualizer.compare_market(15, simulated_market_visualizer, '../results/comparison_real_fake_15_rets.jpg')
real_market_visualizer.compare_market(30, simulated_market_visualizer, '../results/comparison_real_fake_30_rets.jpg')
