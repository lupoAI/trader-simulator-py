import numpy as np
import pandas as pd


class MarketDataAnalyzer:

    def __init__(self, market_data):
        self.market_data = market_data

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
        stylized_facts = pd.DataFrame(index=lags, columns=columns)

        for lag in lags:
            temp = rets.copy()
            temp['Rets Shifted'] = temp['Rets'].shift(lag * returns_interval)
            temp['Rets Sq Shifted'] = temp['Rets Sq'].shift(lag * returns_interval)
            temp['Date Shifted'] = temp['Date'].shift(lag * returns_interval)
            temp = temp.loc[temp['Date'] == temp['Date Shifted']]

            auto_correlation = np.corrcoef(temp['Rets'], temp['Rets Shifted'])
            volatility_clustering = np.corrcoef(temp['Rets Sq'], temp['Rets Sq Shifted'])
            leverage_effect = np.corrcoef(temp['Rets Sq'], temp['Rets Shifted'])

            stylized_facts.at[lag, 'Auto-Correlation'] = auto_correlation[0, 1]
            stylized_facts.at[lag, 'Volatility Clustering'] = volatility_clustering[0, 1]
            stylized_facts.at[lag, 'Leverage Effect'] = leverage_effect[0, 1]

        standard_rets = rets['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = (stylized_facts, density)

        return stylized_facts

    def get_daily_market_metrics(self):
        close = self.market_data[['Close']].groupby(self.market_data.index.date).last()
        close['Rets'] = close / close.shift(1) - 1
        close['Rets Sq'] = close['Rets'] ** 2
        lags = list(range(1, 11))
        columns = ['Auto-Correlation', 'Volatility Clustering', 'Leverage Effect']
        stylized_facts = pd.DataFrame(index=lags, columns=columns)
        for lag in lags:
            ac = np.corrcoef(close['Rets'], close['Rets'].shift(lag))
            vc = np.corrcoef(close['Rets Sq'], close['Rets Sq'].shift(lag))
            le = np.corrcoef(close['Rets Sq'], close['Rets'].shift(lag))

            stylized_facts.at[lag, 'Auto-Correlation'] = ac[0, 1]
            stylized_facts.at[lag, 'Volatility Clustering'] = vc[0, 1]
            stylized_facts.at[lag, 'Leverage Effect'] = le[0, 1]

        standard_rets = close['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = (stylized_facts, density)

        return stylized_facts


class SimulatedMarketAnalyzer:

    def __init__(self, market_prices):
        self.market_prices = market_prices

    def get_market_metrics(self, returns_interval):
        rets = pd.DataFrame(self.market_prices, columns=['Close'])
        rets['Rets'] = rets['Close'] / rets['Close'].shift(returns_interval) - 1
        rets['Rets Sq'] = rets['Rets'] ** 2
        lags = list(range(1, 11))
        columns = ['Auto-Correlation', 'Volatility Clustering', 'Leverage Effect']
        stylized_facts = pd.DataFrame(index=lags, columns=columns)

        for lag in lags:
            temp = rets.copy()
            temp['Rets Shifted'] = temp['Rets'].shift(lag * returns_interval)
            temp['Rets Sq Shifted'] = temp['Rets Sq'].shift(lag * returns_interval)

            auto_correlation = np.corrcoef(temp['Rets'], temp['Rets Shifted'])
            volatility_clustering = np.corrcoef(temp['Rets Sq'], temp['Rets Sq Shifted'])
            leverage_effect = np.corrcoef(temp['Rets Sq'], temp['Rets Shifted'])

            stylized_facts.at[lag, 'Auto-Correlation'] = auto_correlation[0, 1]
            stylized_facts.at[lag, 'Volatility Clustering'] = volatility_clustering[0, 1]
            stylized_facts.at[lag, 'Leverage Effect'] = leverage_effect[0, 1]

        standard_rets = rets['Rets'].dropna().values
        standard_rets = (standard_rets - standard_rets.mean()) / standard_rets.std()
        bins = np.linspace(-50, 50, 100)
        histogram = np.histogram(standard_rets, bins=bins, density=True)
        center_hist = (histogram[1][1:] + histogram[1][:-1]) / 2
        density = pd.Series(histogram[0], index=center_hist)

        stylized_facts = (stylized_facts, density)

        return stylized_facts



if __name__ == '__main__':
    import pickle

    headers = ['Open', 'High', 'Low', 'Close']
    data = pd.read_csv('../data/spx/SPX_1min.txt', header=None, index_col=0, parse_dates=[0])
    data = data.drop(columns=[5])
    data.columns = headers

    market_analyzer = MarketDataAnalyzer(data)
    features_1m = market_analyzer.get_market_metrics(1)
    features_5m = market_analyzer.get_market_metrics(5)
    features_15m = market_analyzer.get_market_metrics(15)
    features_30m = market_analyzer.get_market_metrics(30)
    features_1d = market_analyzer.get_daily_market_metrics()

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
