import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../results/visualize_fcn_gamma/minutes_of_trading.pickle', 'rb') as mins:
    minutes_of_trading = pickle.load(mins)

with open('../results/visualize_fcn_gamma/square_returns.pickle', 'rb') as sq_ret:
    square_returns = pickle.load(sq_ret)

with open('../results/visualize_fcn_gamma/ewma_square_returns.pickle', 'rb') as ewma_sq_ret:
    ewma_square_returns = pickle.load(ewma_sq_ret)

with open('../results/visualize_fcn_gamma/price_series.pickle', 'rb') as ser:
    last_valid_mid_price = pickle.load(ser)

# plt.plot(minutes_of_trading, square_returns)
# plt.show()
#
# plt.plot(minutes_of_trading, ewma_square_returns)
# plt.show()
#
# plt.plot(last_valid_mid_price.price)
# plt.show()

sqrt_ewma = np.sqrt(ewma_square_returns)

plt.plot(minutes_of_trading, sqrt_ewma / 0.004)
plt.show()