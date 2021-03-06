from analysis.market_analyzer import MarketVisualizer, plot_stylized_facts
from market.simulator import RandomSimulator
from market.exchange import Exchange
import pandas as pd
import pickle
import os

if not os.path.exists("../results/visualize_market/"):
    os.mkdir("../results/visualize_market/")

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers

real_market_visualizer = MarketVisualizer(data)
real_market_visualizer.visualize_market(1, '../results/visualize_market/spx_1m_rets_market.jpg')
features_1m = real_market_visualizer.market_analyzer.get_market_metrics(1)
plot_stylized_facts(features_1m)
real_market_visualizer.visualize_market(5, '../results/visualize_market/spx_5m_rets_market.jpg')
features_5m = real_market_visualizer.market_analyzer.get_market_metrics(5)
real_market_visualizer.visualize_market(15, '../results/visualize_market/spx_15m_rets_market.jpg')
features_15m = real_market_visualizer.market_analyzer.get_market_metrics(15)
real_market_visualizer.visualize_market(30, '../results/visualize_market/spx_30m_rets_market.jpg')
features_30m = real_market_visualizer.market_analyzer.get_market_metrics(30)
real_market_visualizer.visualize_market('1d', '../results/visualize_market/spx_1d_rets_market.jpg')
features_1d = real_market_visualizer.market_analyzer.get_daily_market_metrics()
real_market_visualizer.visualize_close_auto_correlation('../results/visualize_market/spx_close_corr.jpg')
features_close = real_market_visualizer.market_analyzer.get_close_auto_correlation()

with open('../data/spx_processed/features_1m.pickle', 'wb') as f_1m:
    pickle.dump(features_1m, f_1m)
with open('../data/spx_processed/features_5m.pickle', 'wb') as f_5m:
    pickle.dump(features_5m, f_5m)
with open('../data/spx_processed/features_15m.pickle', 'wb') as f_15m:
    pickle.dump(features_15m, f_15m)
with open('../data/spx_processed/features_30m.pickle', 'wb') as f_30m:
    pickle.dump(features_30m, f_30m)
with open('../data/spx_processed/features_1d.pickle', 'wb') as f_1d:
    pickle.dump(features_1d, f_1d)

with open('../data/spx_processed/features_close.pickle', 'wb') as f_close:
    pickle.dump(features_close, f_close)

exchange = Exchange()
random_simulator = RandomSimulator(exchange, 100)
random_simulator.run(5000, 20, 500, 5, 20)
simulation_price = random_simulator.last_mid_price_series
simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

simulated_market_visualizer.visualize_market(1, '../results/visualize_market/random_simulation_1m_rets_market.jpg')
simulated_market_visualizer.visualize_market(5, '../results/visualize_market/random_simulation_5m_rets_market.jpg')
simulated_market_visualizer.visualize_market(15, '../results/visualize_market/random_simulation_15m_rets_market.jpg')
simulated_market_visualizer.visualize_market(30, '../results/visualize_market/random_simulation_30m_rets_market.jpg')
