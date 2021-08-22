import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import RandomState
# from pandas.plotting import scatter_matrix

leverage = 3
vol = 0.2
trend = 0
starting_price = 1
T = 252
n_steps = 252 * 4
random_state = RandomState(40)
gauss = random_state.normal(size=(n_steps,))
gbm = np.exp(trend / T + np.sqrt(1 / T) * vol * gauss - 0.5 / T * vol ** 2)

year = np.arange(0, n_steps + 1) / 252
cols = ['Time', 'Underlying Price', 'Underlying Return', 'Target ETF Price', 'Target ETF Return',
        'Exposure Position', 'Shares Held', 'Re-Balancing', 'ETF Replicated']

lev_etf = pd.DataFrame(index=list(range(len(year))), columns=cols)

lev_etf.loc[:, 'Time'] = year

lev_etf.loc[1:, 'Underlying Price'] = starting_price * gbm.cumprod()
lev_etf.loc[0, 'Underlying Price'] = starting_price

lev_etf.loc[:, 'Underlying Return'] = lev_etf['Underlying Price'] / lev_etf['Underlying Price'].shift(1) - 1

lev_etf['Target ETF Return'] = leverage * lev_etf['Underlying Return']
lev_etf.loc[1:, 'Target ETF Price'] = lev_etf.loc[1:, 'Target ETF Return'].values + 1
lev_etf.loc[0, 'Target ETF Price'] = starting_price
lev_etf.loc[:, 'Target ETF Price'] = lev_etf['Target ETF Price'].cumprod()

lev_etf.loc[:, 'Exposure Position'] = lev_etf['Target ETF Price'] * leverage

lev_etf.loc[:, 'Shares Held'] = lev_etf['Exposure Position'] / lev_etf['Underlying Price']

lev_etf.loc[:, 'Re-Balancing'] = lev_etf['Shares Held'] - lev_etf['Shares Held'].shift(1)

lev_etf.loc[0, 'ETF Replicated'] = starting_price
lev_etf.loc[1:, 'ETF Replicated'] = lev_etf['Exposure Position'].shift(1).loc[1:] * lev_etf['Underlying Return'].loc[1:]
lev_etf.loc[:, 'ETF Replicated'] = lev_etf.cumsum()


# plt.plot(lev_etf['Shares Held'], label="shares long way")
# plt.plot(lev_etf['Target ETF Price'] / lev_etf['Underlying Price'] * leverage)
# plt.show()

# plt.plot(lev_etf['Time'], lev_etf['Underlying Price'], label='Underlying Price')
# plt.plot(lev_etf['Time'], lev_etf['ETF Replicated'], label='ETF Replicated')
# plt.grid(True)
# plt.legend()
# plt.show()
#
# plt.scatter(lev_etf['Underlying Return'], lev_etf['Re-Balancing'])
# plt.xlabel('Underlying Return')
# plt.ylabel('Re-Balancing')
# plt.grid(True)
# plt.show()
#
#
# plt.scatter(lev_etf['Underlying Price'], lev_etf['Shares Held'])
# plt.xlabel('Underlying Price')
# plt.ylabel('Shares Held')
# plt.grid(True)
# plt.show()
#
#
# plt.scatter(lev_etf['ETF Replicated'], lev_etf['Shares Held'])
# plt.xlabel('ETF Replicated')
# plt.ylabel('Shares Held')
# plt.grid(True)
# plt.show()

ax1 = plt.subplot2grid(shape=(4, 6), loc=(0, 0), colspan=6, rowspan=2)
ax2 = plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
ax4 = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)

ax1.plot(lev_etf['Time'], lev_etf['Underlying Price'], label='Underlying Price')
ax1.plot(lev_etf['Time'], lev_etf['ETF Replicated'], label='ETF Replicated')
ax1.plot(lev_etf['Time'], lev_etf['Shares Held'] / leverage, label='Shares Held / Leverage')
ax1.grid(True)
ax1.legend()

ax2.scatter(lev_etf['Underlying Return'], lev_etf['Re-Balancing'], color='red')
ax2.set_xlabel('Underlying Return')
ax2.set_ylabel('Re-Balancing')
ax2.grid(True)

ax3.scatter(lev_etf['Underlying Price'], lev_etf['Shares Held'], color='green')
ax3.set_xlabel('Underlying Price')
ax3.set_ylabel('Shares Held')
ax3.grid(True)

ax4.scatter(lev_etf['ETF Replicated'], lev_etf['Shares Held'], color='purple')
ax4.set_xlabel('ETF Price')
ax4.set_ylabel('Shares Held')
ax4.grid(True)

# ax4.scatter(lev_etf['Underlying Price'], lev_etf['Exposure Position'], color='purple')
# ax4.set_xlabel('Underlying Price')
# ax4.set_ylabel('Exposure Position')
# ax4.grid(True)

plt.show()

