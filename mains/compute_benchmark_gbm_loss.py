from analysis.market_analyzer import SimulatedMarketAnalyzer, MarketDataAnalyzer, compare_stylized_facts
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

n_series = 2
n_steps = 10000
T = 10000
vol = 0.2
trend = - 0.5 * vol ** 2
random_seed = 100
starting_price = 10000
save = False
show_samples = True

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
    for i, fact in enumerate(facts):
        if i == 0 and show_samples:
            compare_stylized_facts(target_facts, fact, f"../results/compute_benchmark_gbm_loss/sample_{lag}_rets.jpg")
        loss = LossFunction(target_facts, fact)
        loss.compute_loss()
        losses = losses.append(loss.to_df(), ignore_index=True)

    if save:
        losses.to_csv(f"../results/compute_benchmark_gbm_loss/benchmark_loss_{lag}_rets.csv", index=False)


columns = ["auto_correlation_loss", "volatility_clustering_loss", "leverage_effect_loss", "distribution_loss",
           "total_loss"]

mean_loss = pd.DataFrame(index=list_rets, columns=columns)
std_loss = pd.DataFrame(index=list_rets, columns=columns)
for lag in list_rets:
    benchmark = pd.read_csv(f"../results/compute_benchmark_gbm_loss/benchmark_loss_{lag}_rets.csv")
    mean_loss.loc[lag] = benchmark.mean(axis=0)
    std_loss.loc[lag] = benchmark.std(axis=0)

mean_loss.index.name = "rets"
std_loss.index.name = "rets"

if save:
    mean_loss.to_csv(f"../results/compute_benchmark_gbm_loss/benchmark_mean_loss.csv")
    std_loss.to_csv(f"../results/compute_benchmark_gbm_loss/benchmark_std_loss.csv")
