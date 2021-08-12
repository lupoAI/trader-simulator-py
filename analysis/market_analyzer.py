import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
import os
from scipy.stats import wasserstein_distance


class StylizedFacts:

    def __init__(self, correlation: pd.DataFrame, density: pd.Series, rets: np.array):
        self.auto_correlation = correlation['Auto-Correlation']
        self.volatility_clustering = correlation['Volatility Clustering']
        self.leverage_effect = correlation['Leverage Effect']
        self.density = density
        self.rets = rets


def compare_stylized_facts(stylized_facts: StylizedFacts, other_stylized_facts: StylizedFacts,
                           save_name: Union[str, None] = None):
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)
    ax1.plot(stylized_facts.auto_correlation, color='red', label='target')
    ax1.plot(other_stylized_facts.auto_correlation, color='blue', label='simulation')
    ax1.grid(True)
    ax1.set_xticks(list(range(1, 11)))
    ax1.set_title('Auto-Correlation', size=10)
    ax1.legend(fontsize=8)
    ax2.plot(stylized_facts.volatility_clustering, color='orange', label='target')
    ax2.plot(other_stylized_facts.volatility_clustering, color='green', label='simulation')
    ax2.grid(True)
    ax2.set_xticks(list(range(1, 11)))
    ax2.set_title('Volatility Clustering', size=10)
    ax2.legend(fontsize=8)
    ax3.plot(stylized_facts.leverage_effect, color='purple', label='target')
    ax3.plot(other_stylized_facts.leverage_effect, color='brown', label='simulation')
    ax3.grid(True)
    ax3.set_xticks(list(range(1, 11)))
    ax3.set_title('Leverage Effect', size=10)
    ax3.legend(fontsize=8)
    bins = np.linspace(-50, 50, 1000)
    centers = (bins[1:] + bins[:-1]) / 2
    normal_distribution = normal_pdf(centers)
    ax4.hist(stylized_facts.rets, bins=bins, density=True, label='target', alpha=0.5)
    ax4.hist(other_stylized_facts.rets, bins=bins, density=True, label='simulation', alpha=0.5)
    ax4.plot(centers, normal_distribution, label='Normal PDF')
    ax4.set_xlim(-5, 5)
    ax4.set_xticks(list(range(-5, 6)))
    ax4.set_title('Return Distribution', size=10)
    ax4.legend(fontsize=8)
    plt.subplots_adjust(hspace=0.55, wspace=1)
    plt.suptitle(f'Markets comparison')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


def plot_stylized_facts(stylized_facts: StylizedFacts, save_name: Union[str, None] = None):
    ax1 = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)
    ax1.plot(stylized_facts.auto_correlation, color='red')
    ax1.grid(True)
    ax1.set_xticks(list(range(1, 11)))
    ax1.set_title('Auto-Correlation', size=10)
    ax1.legend(fontsize=8)
    ax2.plot(stylized_facts.volatility_clustering, color='orange')
    ax2.grid(True)
    ax2.set_xticks(list(range(1, 11)))
    ax2.set_title('Volatility Clustering', size=10)
    ax2.legend(fontsize=8)
    ax3.plot(stylized_facts.leverage_effect, color='purple')
    ax3.grid(True)
    ax3.set_xticks(list(range(1, 11)))
    ax3.set_title('Leverage Effect', size=10)
    ax3.legend(fontsize=8)
    bins = np.linspace(-50, 50, 1000)
    centers = (bins[1:] + bins[:-1]) / 2
    normal_distribution = normal_pdf(centers)
    ax4.hist(stylized_facts.rets, bins=bins, density=True, alpha=0.5)
    ax4.plot(centers, normal_distribution, label='Normal PDF')
    ax4.set_xlim(-5, 5)
    ax4.set_xticks(list(range(-5, 6)))
    ax4.set_title('Return Distribution', size=10)
    ax4.legend(fontsize=8)
    plt.subplots_adjust(hspace=0.55, wspace=1)
    plt.suptitle(f'Stylized Facts')
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


class MarketDataAnalyzer:

    def __init__(self, market_data):
        self.market_data = market_data

    def __del__(self):
        del self.market_data
        del self

    def get_market_metrics(self, returns_interval):

        if os.path.exists(f"../data/spx_processed/features_{returns_interval}m.pickle"):
            with open(f"../data/spx_processed/features_{returns_interval}m.pickle", 'rb') as features:
                stylized_facts = pickle.load(features)
            return stylized_facts

        rets = self.market_data.copy().drop(columns=['Open', 'High', 'Low'])
        rets['Date'] = rets.index.date
        rets['Date Shifted'] = rets['Date'].shift(returns_interval)
        rets['Rets'] = rets['Close'] / rets['Close'].shift(returns_interval) - 1
        rets['Rets Sq'] = rets['Rets'] ** 2
        # Drop extra-day returns
        rets = rets.loc[rets['Date'] == rets['Date Shifted']]
        rets = rets.drop(columns=['Date Shifted'])
        lags = list(range(1, 11))
        columns = ['Auto-Correlation', 'Volatility Clustering', 'Leverage Effect']
        correlation_facts = pd.DataFrame(index=lags, columns=columns)

        for lag in lags:
            temp = rets.copy()
            temp['Rets Shifted'] = temp['Rets'].shift(lag * returns_interval)
            temp['Rets Sq Shifted'] = temp['Rets Sq'].shift(lag * returns_interval)
            temp['Date Shifted'] = temp['Date'].shift(lag * returns_interval)
            temp = temp.loc[temp['Date'] == temp['Date Shifted']]
            temp = temp.dropna()

            auto_correlation = np.corrcoef(temp['Rets'], temp['Rets Shifted'])
            volatility_clustering = np.corrcoef(temp['Rets Sq'], temp['Rets Sq Shifted'])
            leverage_effect = np.corrcoef(temp['Rets Sq'], temp['Rets Shifted'])

            correlation_facts.at[lag, 'Auto-Correlation'] = auto_correlation[0, 1]
            correlation_facts.at[lag, 'Volatility Clustering'] = volatility_clustering[0, 1]
            correlation_facts.at[lag, 'Leverage Effect'] = leverage_effect[0, 1]

        standard_rets = rets['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = StylizedFacts(correlation_facts, density, standard_rets)

        return stylized_facts

    def get_daily_market_metrics(self):
        if os.path.exists("../data/spx_processed/features_1d.pickle"):
            with open("../data/spx_processed/features_1d.pickle", 'rb') as features:
                stylized_facts = pickle.load(features)
            return stylized_facts

        close = self.market_data[['Close']].groupby(self.market_data.index.date).last()
        close['Rets'] = close / close.shift(1) - 1
        close['Rets Sq'] = close['Rets'] ** 2
        lags = list(range(1, 11))
        columns = ['Auto-Correlation', 'Volatility Clustering', 'Leverage Effect']
        correlation_facts = pd.DataFrame(index=lags, columns=columns)
        for lag in lags:
            temp = close.copy()
            temp['Rets Shifted'] = temp['Rets'].shift(lag)
            temp['Rets Sq Shifted'] = temp['Rets Sq'].shift(lag)
            temp = temp.dropna()

            auto_correlation = np.corrcoef(temp['Rets'], temp['Rets Shifted'])
            volatility_clustering = np.corrcoef(temp['Rets Sq'], temp['Rets Sq Shifted'])
            leverage_effect = np.corrcoef(temp['Rets Sq'], temp['Rets Shifted'])

            correlation_facts.at[lag, 'Auto-Correlation'] = auto_correlation[0, 1]
            correlation_facts.at[lag, 'Volatility Clustering'] = volatility_clustering[0, 1]
            correlation_facts.at[lag, 'Leverage Effect'] = leverage_effect[0, 1]

        standard_rets = close['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = StylizedFacts(correlation_facts, density, standard_rets)

        return stylized_facts

    def get_close_auto_correlation(self):

        if os.path.exists("../data/spx_processed/features_close.pickle"):
            with open("../data/spx_processed/features_close.pickle", 'rb') as features:
                corr = pickle.load(features)
            return corr

        close = self.market_data[['Close']].copy()
        close.loc[:, 'Time'] = close.index.time
        close.loc[:, 'Date'] = close.index.date
        start_time = 9 * 60 + 30
        length_of_trading_day = 390
        today = datetime.datetime(1900, 1, 1)
        unique_times = [today + datetime.timedelta(minutes=start_time + m) for m in range(length_of_trading_day + 1)]
        unique_times = np.array([x.time() for x in unique_times])
        n_times = len(unique_times) - 1
        idx = [0, int(n_times / 2), int(6 * n_times / 10), int(7 * n_times / 10), int(8 * n_times / 10),
               int(9 * n_times / 10), n_times]
        times = unique_times[idx]
        start_time = times[0]
        end_time = times[-1]
        correlations = []
        for mid_time in times[1:-1]:
            kwargs = {'start': start_time, 'end': end_time, 'mid': mid_time}
            rets = close.groupby('Date').apply(get_corr_returns, **kwargs).dropna()
            get_first = np.vectorize(lambda x: x[0])
            get_second = np.vectorize(lambda x: x[1])
            first = get_first(rets.values)
            second = get_second(rets.values)
            correlations += [np.corrcoef(first, second)[0, 1]]

        return np.array(correlations)


def get_corr_returns(arr, start, end, mid):
    price_st = arr['Close'].loc[arr['Time'] == start]
    price_en = arr['Close'].loc[arr['Time'] == end]
    price_md = arr['Close'].loc[arr['Time'] == mid]

    if len(price_st) == 0 or len(price_en) == 0 or len(price_md) == 0:
        return np.nan

    return [price_md[0] / price_st[0] - 1, price_en[0] / price_md[0] - 1]


class SimulatedMarketAnalyzer:

    def __init__(self, market_prices):
        self.market_prices = market_prices

    def __del__(self):
        del self.market_prices
        del self

    def get_market_metrics(self, returns_interval):
        rets = pd.DataFrame(self.market_prices, columns=['Close'])
        rets['Rets'] = rets['Close'] / rets['Close'].shift(returns_interval) - 1
        rets['Rets Sq'] = rets['Rets'] ** 2
        lags = list(range(1, 11))
        columns = ['Auto-Correlation', 'Volatility Clustering', 'Leverage Effect']
        correlation_facts = pd.DataFrame(index=lags, columns=columns)

        for lag in lags:
            temp = rets.copy()
            temp['Rets Shifted'] = temp['Rets'].shift(lag * returns_interval)
            temp['Rets Sq Shifted'] = temp['Rets Sq'].shift(lag * returns_interval)
            temp = temp.dropna()

            auto_correlation = np.corrcoef(temp['Rets'], temp['Rets Shifted'])
            volatility_clustering = np.corrcoef(temp['Rets Sq'], temp['Rets Sq Shifted'])
            leverage_effect = np.corrcoef(temp['Rets Sq'], temp['Rets Shifted'])

            correlation_facts.at[lag, 'Auto-Correlation'] = auto_correlation[0, 1]
            correlation_facts.at[lag, 'Volatility Clustering'] = volatility_clustering[0, 1]
            correlation_facts.at[lag, 'Leverage Effect'] = leverage_effect[0, 1]

        standard_rets = rets['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = StylizedFacts(correlation_facts, density, standard_rets)

        return stylized_facts

    def get_daily_market_metrics(self):
        rets = pd.DataFrame(self.market_prices, columns=['Close'])
        rets = rets.iloc[::390]
        rets['Rets'] = rets['Close'] / rets['Close'].shift(1) - 1
        rets['Rets Sq'] = rets['Rets'] ** 2
        lags = list(range(1, 11))
        columns = ['Auto-Correlation', 'Volatility Clustering', 'Leverage Effect']
        correlation_facts = pd.DataFrame(index=lags, columns=columns)

        for lag in lags:
            temp = rets.copy()
            temp['Rets Shifted'] = temp['Rets'].shift(lag)
            temp['Rets Sq Shifted'] = temp['Rets Sq'].shift(lag)
            temp = temp.dropna()

            auto_correlation = np.corrcoef(temp['Rets'], temp['Rets Shifted'])
            volatility_clustering = np.corrcoef(temp['Rets Sq'], temp['Rets Sq Shifted'])
            leverage_effect = np.corrcoef(temp['Rets Sq'], temp['Rets Shifted'])

            correlation_facts.at[lag, 'Auto-Correlation'] = auto_correlation[0, 1]
            correlation_facts.at[lag, 'Volatility Clustering'] = volatility_clustering[0, 1]
            correlation_facts.at[lag, 'Leverage Effect'] = leverage_effect[0, 1]

        standard_rets = rets['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = StylizedFacts(correlation_facts, density, standard_rets)

        return stylized_facts

    def get_close_auto_correlation(self):
        close = pd.DataFrame(self.market_prices, columns=['Close'])
        n_times = 390
        idx = [0, int(n_times / 2), int(6 * n_times / 10), int(7 * n_times / 10), int(8 * n_times / 10),
               int(9 * n_times / 10), n_times - 1]
        start = close['Close'].iloc[idx[0]::n_times]
        end = close['Close'].iloc[idx[-1]::n_times]
        correlations = []
        for i in idx[1:-1]:
            mid = close['Close'].iloc[i::n_times]
            len_seg_1 = min(len(start), len(mid))
            len_seg_2 = min(len(mid), len(end))
            first = mid.iloc[:len_seg_1].values / start.iloc[:len_seg_1].values - 1
            second = end.iloc[:len_seg_2].values / mid.iloc[:len_seg_2].values - 1
            len_fin = min(len(first), len(second))
            first = first[:len_fin]
            second = second[:len_fin]
            mask = ~np.isnan(first) & ~np.isnan(second)
            first = first[mask]
            second = second[mask]
            correlations += [np.corrcoef(first, second)[0, 1]]

        return np.array(correlations)


class MarketVisualizerAbstract:
    def __init__(self, market_prices, is_simulated=False):
        self.market_prices = market_prices
        self.is_simulated = is_simulated
        if not is_simulated:
            self.market_analyzer = MarketDataAnalyzer(self.market_prices)
            self.market_prices = self.market_prices['Close']
        else:
            self.market_analyzer = SimulatedMarketAnalyzer(self.market_prices)
            self.market_prices = np.array(self.market_prices, dtype=np.float)

    def __del__(self):
        del self.market_prices
        del self.market_analyzer
        del self.is_simulated
        del self


class MarketVisualizer(MarketVisualizerAbstract):

    def __init__(self, market_prices, is_simulated=False):
        super().__init__(market_prices, is_simulated)

    def visualize_market(self, returns_interval: Union[int, str], save_name: Union[str, None] = None):
        if returns_interval == '1d':
            stylized_facts = self.market_analyzer.get_daily_market_metrics()
        else:
            stylized_facts = self.market_analyzer.get_market_metrics(returns_interval)
        ax1 = plt.subplot2grid(shape=(3, 6), loc=(0, 0), colspan=6)
        ax2 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
        ax4 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
        ax5 = plt.subplot2grid((3, 6), (2, 0), colspan=6)
        ax1.plot(self.market_prices, color='green')
        ax1.grid(True)
        ax1.set_title('Price', size=10)
        ax2.plot(stylized_facts.auto_correlation, color='red')
        ax2.grid(True)
        ax2.set_xticks(list(range(1, 11)))
        ax2.set_title('Auto-Correlation', size=10)
        ax3.plot(stylized_facts.volatility_clustering, color='orange')
        ax3.grid(True)
        ax3.set_xticks(list(range(1, 11)))
        ax3.set_title('Volatility Clustering', size=10)
        ax4.plot(stylized_facts.leverage_effect, color='purple')
        ax4.grid(True)
        ax4.set_xticks(list(range(1, 11)))
        ax4.set_title('Leverage Effect', size=10)
        bins = np.linspace(-50, 50, 1000)
        centers = (bins[1:] + bins[:-1]) / 2
        normal_distribution = normal_pdf(centers)
        ax5.hist(stylized_facts.rets, bins=bins, density=True)
        ax5.plot(centers, normal_distribution, label='Normal PDF')
        ax5.set_xlim(-5, 5)
        ax5.set_xticks(list(range(-5, 6)))
        ax5.set_title('Return Distribution', size=10)
        plt.subplots_adjust(hspace=0.55, wspace=1)
        if self.is_simulated:
            plt.suptitle(f'Simulated Market facts for {returns_interval} periods returns')
        else:
            plt.suptitle(f'SPX facts for {returns_interval} periods returns')
        plt.legend()
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()

    def compare_market(self, returns_interval: Union[int, str], other: MarketVisualizerAbstract,
                       save_name: Union[str, None] = None):
        if returns_interval == '1d':
            stylized_facts = self.market_analyzer.get_daily_market_metrics()
            other_stylized_facts = other.market_analyzer.get_daily_market_metrics()
        else:
            stylized_facts = self.market_analyzer.get_market_metrics(returns_interval)
            other_stylized_facts = other.market_analyzer.get_market_metrics(returns_interval)
        ax1 = plt.subplot2grid(shape=(4, 6), loc=(0, 0), colspan=6)
        ax1_1 = ax1.twiny()
        ax2 = plt.subplot2grid((4, 6), (1, 0), colspan=3)
        ax3 = plt.subplot2grid((4, 6), (1, 3), colspan=3)
        ax4 = plt.subplot2grid((4, 6), (2, 3), colspan=3)
        ax5 = plt.subplot2grid((4, 6), (3, 0), colspan=6)
        ax6 = plt.subplot2grid((4, 6), (2, 0), colspan=3)
        scaled_market_prices = self.market_prices / self.market_prices[0]
        first_non_nan = other.market_prices[~np.isnan(other.market_prices)][0]
        scaled_other_market_prices = other.market_prices / first_non_nan
        ax1.plot(scaled_market_prices, color='green', label='Price target')
        ax1_1.plot(scaled_other_market_prices, color='red', label='Price simulation')
        ax1.grid(True)
        ax1.legend(fontsize=8)
        ax1_1.legend(fontsize=8)
        ax2.plot(stylized_facts.auto_correlation, color='red', label='target')
        ax2.plot(other_stylized_facts.auto_correlation, color='blue', label='simulation')
        ax2.grid(True)
        ax2.set_xticks(list(range(1, 11)))
        ax2.set_title('Auto-Correlation', size=10)
        ax2.legend(fontsize=8)
        ax3.plot(stylized_facts.volatility_clustering, color='orange', label='target')
        ax3.plot(other_stylized_facts.volatility_clustering, color='green', label='simulation')
        ax3.grid(True)
        ax3.set_xticks(list(range(1, 11)))
        ax3.set_title('Volatility Clustering', size=10)
        ax3.legend(fontsize=8)
        ax4.plot(stylized_facts.leverage_effect, color='purple', label='target')
        ax4.plot(other_stylized_facts.leverage_effect, color='brown', label='simulation')
        ax4.grid(True)
        ax4.set_xticks(list(range(1, 11)))
        ax4.set_title('Leverage Effect', size=10)
        ax4.legend(fontsize=8)
        bins = np.linspace(-50, 50, 1000)
        centers = (bins[1:] + bins[:-1]) / 2
        normal_distribution = normal_pdf(centers)
        ax5.hist(stylized_facts.rets, bins=bins, density=True, label='target', alpha=0.5)
        ax5.hist(other_stylized_facts.rets, bins=bins, density=True, label='simulation', alpha=0.5)
        ax5.plot(centers, normal_distribution, label='Normal PDF')
        ax5.set_xlim(-5, 5)
        ax5.set_xticks(list(range(-5, 6)))
        ax5.set_title('Return Distribution', size=10)
        ax5.legend(fontsize=8)
        loss = LossFunction(stylized_facts, other_stylized_facts)
        loss.compute_loss()
        labels = ['$l^{ac}$', '$l^{vc}$', '$l^{le}$', '$l^{ft}$']
        colors = ['blue', 'red', 'green', 'orange']
        values = [loss.auto_correlation_loss, loss.volatility_clustering_loss, loss.leverage_effect_loss,
                  loss.distribution_loss]
        ax6.barh(list(range(4)), values, tick_label=labels, color=colors)
        ax6.set_title("Losses", size=10)
        ax6.grid(True)
        plt.subplots_adjust(hspace=0.70, wspace=1)
        plt.suptitle(f'Markets comparison for {returns_interval} periods returns')
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()


    def visualize_close_auto_correlation(self, save_name: Union[str, None] = None):
        target_close_correlation = self.market_analyzer.get_close_auto_correlation()
        ticks = list(range(1, 6))
        labels = ['$\\frac{1}{2}$', '$\\frac{6}{10}$', '$\\frac{7}{10}$', '$\\frac{8}{10}$', '$\\frac{9}{10}$']

        plt.plot(ticks, target_close_correlation)
        plt.title('Close Correlation')
        plt.ylabel('Correlation')
        plt.xticks(ticks, labels)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()

    def compare_close_auto_correlation(self, other: MarketVisualizerAbstract, save_name: Union[str, None] = None):
        target_close_correlation = self.market_analyzer.get_close_auto_correlation()
        simulated_close_correlation = other.market_analyzer.get_close_auto_correlation()
        ticks = list(range(1, 6))
        labels = ['$\\frac{1}{2}$', '$\\frac{6}{10}$', '$\\frac{7}{10}$', '$\\frac{8}{10}$', '$\\frac{9}{10}$']

        plt.plot(ticks, target_close_correlation, label='target')
        plt.plot(ticks, simulated_close_correlation, label='simulated')
        plt.legend()
        plt.title('Close Correlation')
        plt.xlabel('Correlation')
        plt.xticks(ticks, labels)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()


class LossFunction:

    def __init__(self, target_facts: StylizedFacts, simulated_facts: StylizedFacts):
        self.target_facts = target_facts
        self.simulated_facts = simulated_facts
        self.auto_correlation_loss = None
        self.volatility_clustering_loss = None
        self.leverage_effect_loss = None
        self.distribution_loss = None
        self.total_loss = None

    def compute_loss(self):
        if self.auto_correlation_loss is None:
            self.compute_auto_correlation_loss()
        if self.volatility_clustering_loss is None:
            self.compute_volatility_clustering_loss()
        if self.leverage_effect_loss is None:
            self.compute_leverage_effect_loss()
        if self.distribution_loss is None:
            self.compute_distribution_loss()
        total_loss = 0
        total_loss += self.auto_correlation_loss
        total_loss += self.volatility_clustering_loss
        total_loss += self.leverage_effect_loss
        total_loss += self.distribution_loss
        total_loss /= 4
        self.total_loss = total_loss
        return total_loss

    def compute_auto_correlation_loss(self):
        target = self.target_facts.auto_correlation
        simulation = self.simulated_facts.auto_correlation
        loss = np.abs(target.values - simulation.values).mean()
        self.auto_correlation_loss = loss

    def compute_volatility_clustering_loss(self):
        target = self.target_facts.volatility_clustering
        simulation = self.simulated_facts.volatility_clustering
        loss = np.abs(target.values - simulation.values).mean()
        self.volatility_clustering_loss = loss

    def compute_leverage_effect_loss(self):
        target = self.target_facts.leverage_effect
        simulation = self.simulated_facts.leverage_effect
        loss = np.abs(target.values - simulation.values).mean()
        self.leverage_effect_loss = loss

    def compute_distribution_loss(self):
        # target = self.target_facts.density
        # simulation = self.simulated_facts.density
        # loss = wasserstein_distance(target.index.values, simulation.index.values,
        #                             target.values, simulation.values)
        target = self.target_facts.rets
        simulation = self.simulated_facts.rets
        loss = wasserstein_distance(target, simulation)
        self.distribution_loss = loss

    def to_df(self):
        columns = ["auto_correlation_loss", "volatility_clustering_loss",
                   "leverage_effect_loss", "distribution_loss", "total_loss"]
        values = [[self.auto_correlation_loss, self.volatility_clustering_loss,
                   self.leverage_effect_loss, self.distribution_loss, self.total_loss]]

        return pd.DataFrame(values, columns=columns)


def normal_pdf(values):
    return np.exp(-values ** 2 / 2) / np.sqrt(2 * np.pi)
