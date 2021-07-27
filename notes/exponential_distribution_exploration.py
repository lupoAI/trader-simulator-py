import numpy as np
import matplotlib.pyplot as plt

average_trade_per_step = 3

waiting_times = np.random.exponential(1/average_trade_per_step, size=100000)

time_of_trade = waiting_times.cumsum()

minute_of_trade = time_of_trade.astype(int)

values, counts = np.unique(minute_of_trade, return_counts=True)

plt.hist(counts)
plt.show()