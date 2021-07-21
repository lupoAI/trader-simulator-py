import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union


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
    ax1.plot(stylized_facts.auto_correlation, color='red', label='1')
    ax1.plot(other_stylized_facts.auto_correlation, color='blue', label='2')
    ax1.grid(True)
    ax1.set_xticks(list(range(1, 11)))
    ax1.set_title('Auto-Correlation', size=10)
    ax1.legend(fontsize=8)
    ax2.plot(stylized_facts.volatility_clustering, color='orange', label='1')
    ax2.plot(other_stylized_facts.volatility_clustering, color='green', label='2')
    ax2.grid(True)
    ax2.set_xticks(list(range(1, 11)))
    ax2.set_title('Volatility Clustering', size=10)
    ax2.legend(fontsize=8)
    ax3.plot(stylized_facts.leverage_effect, color='purple', label='1')
    ax3.plot(other_stylized_facts.leverage_effect, color='brown', label='2')
    ax3.grid(True)
    ax3.set_xticks(list(range(1, 11)))
    ax3.set_title('Leverage Effect', size=10)
    ax3.legend(fontsize=8)
    bins = np.linspace(-50, 50, 1000)
    centers = (bins[1:] + bins[:-1]) / 2
    normal_distribution = normal_pdf(centers)
    ax4.hist(stylized_facts.rets, bins=bins, density=True, label='1', alpha=0.5)
    ax4.hist(other_stylized_facts.rets, bins=bins, density=True, label='2', alpha=0.5)
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


class MarketDataAnalyzer:

    def __init__(self, market_data):
        self.market_data = market_data

    def __del__(self):
        del self.market_data
        del self

    def get_market_metrics(self, returns_interval):
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
        if returns_interval == '1d' and not self.is_simulated:
            stylized_facts = self.market_analyzer.get_daily_market_metrics()
        elif returns_interval == '1d' and self.is_simulated:
            raise ValueError('1d is only available for real market data')
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
        if returns_interval == '1d' and not self.is_simulated:
            stylized_facts = self.market_analyzer.get_daily_market_metrics()
        elif returns_interval == '1d' and self.is_simulated:
            raise ValueError('1d is only available for real market data')
        else:
            stylized_facts = self.market_analyzer.get_market_metrics(returns_interval)
        if returns_interval == '1d' and not other.is_simulated:
            other_stylized_facts = other.market_analyzer.get_daily_market_metrics()
        elif returns_interval == '1d' and other.is_simulated:
            raise ValueError('1d is only available for real market data')
        else:
            other_stylized_facts = other.market_analyzer.get_market_metrics(returns_interval)

        ax1 = plt.subplot2grid(shape=(3, 6), loc=(0, 0), colspan=6)
        ax1_1 = ax1.twiny()
        ax2 = plt.subplot2grid((3, 6), (1, 0), colspan=2)
        ax3 = plt.subplot2grid((3, 6), (1, 2), colspan=2)
        ax4 = plt.subplot2grid((3, 6), (1, 4), colspan=2)
        ax5 = plt.subplot2grid((3, 6), (2, 0), colspan=6)
        scaled_market_prices = self.market_prices / self.market_prices[0]
        first_non_nan = other.market_prices[~np.isnan(other.market_prices)][0]
        scaled_other_market_prices = other.market_prices / first_non_nan
        ax1.plot(scaled_market_prices, color='green', label='Price 1')
        ax1_1.plot(scaled_other_market_prices, color='yellow', label='Price 2')
        ax1.grid(True)
        ax1.legend(fontsize=8)
        ax1_1.legend(fontsize=8)
        ax2.plot(stylized_facts.auto_correlation, color='red', label='1')
        ax2.plot(other_stylized_facts.auto_correlation, color='blue', label='2')
        ax2.grid(True)
        ax2.set_xticks(list(range(1, 11)))
        ax2.set_title('Auto-Correlation', size=10)
        ax2.legend(fontsize=8)
        ax3.plot(stylized_facts.volatility_clustering, color='orange', label='1')
        ax3.plot(other_stylized_facts.volatility_clustering, color='green', label='2')
        ax3.grid(True)
        ax3.set_xticks(list(range(1, 11)))
        ax3.set_title('Volatility Clustering', size=10)
        ax3.legend(fontsize=8)
        ax4.plot(stylized_facts.leverage_effect, color='purple', label='1')
        ax4.plot(other_stylized_facts.leverage_effect, color='brown', label='2')
        ax4.grid(True)
        ax4.set_xticks(list(range(1, 11)))
        ax4.set_title('Leverage Effect', size=10)
        ax4.legend(fontsize=8)
        bins = np.linspace(-50, 50, 1000)
        centers = (bins[1:] + bins[:-1]) / 2
        normal_distribution = normal_pdf(centers)
        ax5.hist(stylized_facts.rets, bins=bins, density=True, label='1', alpha=0.5)
        ax5.hist(other_stylized_facts.rets, bins=bins, density=True, label='2', alpha=0.5)
        ax5.plot(centers, normal_distribution, label='Normal PDF')
        ax5.set_xlim(-5, 5)
        ax5.set_xticks(list(range(-5, 6)))
        ax5.set_title('Return Distribution', size=10)
        ax5.legend(fontsize=8)
        plt.subplots_adjust(hspace=0.55, wspace=1)
        plt.suptitle(f'Markets comparison for {returns_interval} periods returns')
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()


def normal_pdf(values):
    return np.exp(-values ** 2 / 2) / np.sqrt(2 * np.pi)
