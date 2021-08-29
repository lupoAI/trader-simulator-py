import numpy as np
import matplotlib.pyplot as plt

random_noise = np.random.normal(size=1000)

rets = []
square_rets = []
mean = 0
var = 0
std = 0
mean_list = []
std_list = []
ewma_std_list = []
n = 1
lam = 0.02
ewma_var = 0
ewma_std = 0
for i, ret in enumerate(random_noise):
    rets += [ret]
    square_rets += [ret ** 2]
    prev_mean = mean

    mean += (ret - mean) / n

    var += ((ret - mean) ** 2 - var) / n + (n - 1) / n * ((mean - prev_mean) ** 2)
    std = np.sqrt(var)

    ewma_var += lam * ((ret - mean) ** 2 - ewma_var)
    ewma_std = np.sqrt(ewma_var)

    mean_list += [mean]
    std_list += [std]
    ewma_std_list += [ewma_std]

    np_mean = np.mean(rets)
    np_std = np.std(rets)

    assert np.isclose(mean, np_mean)
    assert np.isclose(std, np_std)

    n += 1

mean_list = np.array(mean_list)
std_list = np.array(std_list)
ewma_std_list = np.array(ewma_std_list)

plt.plot(rets)
plt.plot(mean_list, label='mean')
plt.plot(std_list, label='std')
plt.plot(ewma_std_list, label='ewma std')
plt.legend()
plt.show()

plt.plot(ewma_std_list / std_list, label='proportion')
plt.plot(np.exp(ewma_std_list / std_list - 1), label=' exp proportion')
plt.plot(np.exp(4 * (ewma_std_list / std_list - 1)), label='geared exp proportion')
plt.legend()
plt.show()