from analysis.market_analyzer import MarketVisualizer
from market.simulator import RandomSimulator
from market.exchange import Exchange
import pandas as pd
import pickle

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers

real_market_visualizer = MarketVisualizer(data)
real_market_visualizer.visualize_market(1, '../results/spx_1m_rets_market.jpg')
features_1m = real_market_visualizer.market_analyzer.get_market_metrics(1)
real_market_visualizer.visualize_market(5, '../results/spx_5m_rets_market.jpg')
features_5m = real_market_visualizer.market_analyzer.get_market_metrics(5)
real_market_visualizer.visualize_market(15, '../results/spx_15m_rets_market.jpg')
features_15m = real_market_visualizer.market_analyzer.get_market_metrics(15)
real_market_visualizer.visualize_market(30, '../results/spx_30m_rets_market.jpg')
features_30m = real_market_visualizer.market_analyzer.get_market_metrics(30)
real_market_visualizer.visualize_market('1d', '../results/spx_1d_rets_market.jpg')
features_1d = real_market_visualizer.market_analyzer.get_daily_market_metrics()

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


exchange = Exchange()
random_simulator = RandomSimulator(exchange, 100)
random_simulator.run(5000, 20, 500, 5, 20)
simulation_price = random_simulator.last_mid_price_series
simulated_market_visualizer = MarketVisualizer(simulation_price.price, is_simulated=True)

simulated_market_visualizer.visualize_market(1, '../results/random_simulation_1m_rets_market.jpg')
simulated_market_visualizer.visualize_market(5, '../results/random_simulation_5m_rets_market.jpg')
simulated_market_visualizer.visualize_market(15, '../results/random_simulation_15m_rets_market.jpg')
simulated_market_visualizer.visualize_market(30, '../results/random_simulation_30m_rets_market.jpg')