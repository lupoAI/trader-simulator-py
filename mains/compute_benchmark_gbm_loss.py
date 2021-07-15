from analysis.market_analyzer import SimulatedMarketAnalyzer, MarketDataAnalyzer
from analysis.loss_function import LossFunction
import numpy as np
from numpy.random import RandomState
import pandas as pd
from tqdm import tqdm
import os

list_rets = [1, 5, 15, 30, 7 * 60]

if not os.path.exists("../results/compute_benchmark_gbm_loss/"):
    os.mkdir("../results/compute_benchmark_gbm_loss/")

headers = ['Open', 'High', 'Low', 'Close']
data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
data = data.drop(columns=[5])
data.columns = headers
real_market_analyzer = MarketDataAnalyzer(data)

n_series = 300
n_steps = 10000
T = 10000
vol = 0.2
trend = 0
random_seed = 100
starting_price = 10000

random_state = RandomState(random_seed)
gauss = random_state.normal(size=(n_steps, n_series))
gmb = np.exp(trend / T + np.sqrt(1 / T) * vol * gauss)
gmb = starting_price * gmb.cumprod(axis=0)

time_series = []
for i in tqdm(range(n_series)):
    series = pd.Series(gmb[:, i], name='Close')
    time_series += [SimulatedMarketAnalyzer(series)]

stylized_facts = {}
for lag in tqdm(list_rets):
    stylized_facts[lag] = [x.get_market_metrics(lag) for x in time_series]

for lag in tqdm(list_rets):
    losses = pd.DataFrame()
    facts = stylized_facts[lag]
    if lag == 7 * 60:
        target_facts = real_market_analyzer.get_daily_market_metrics()
    else:
        target_facts = real_market_analyzer.get_market_metrics(lag)
    for fact in facts:
        loss = LossFunction(target_facts, fact)
        loss.compute_loss()
        losses = losses.append(loss.to_df(), ignore_index=True)

    losses.to_csv(f"../results/compute_benchmark_gbm_loss/benchmark_loss_{lag}_rets.csv", index=False)
