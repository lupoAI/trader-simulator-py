import numpy as np
import pandas as pd
from numpy.random import RandomState

leverage = 2
vol = 0.2
trend = 0
starting_price = 1
T = 252
n_steps = 252 * 4
random_state = RandomState(1001)
gauss = random_state.normal(size=(n_steps, ))
gbm = np.exp(trend / T + np.sqrt(1 / T) * vol * gauss - 0.5 / T * vol ** 2)

year = np.arange(0, n_steps + 1 ) / 252
cols = ['Underlying Price', 'Underlying Return', 'Target ETF Price', 'Target ETF Return',
        'Exposure Position', 'Shares Held', 'Rebalancing', 'ETF Replicated']

leveraged_etf_info = pd.DataFrame(index = year, columns=)



