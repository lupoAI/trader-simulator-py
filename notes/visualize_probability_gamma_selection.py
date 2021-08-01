import numpy as np
import matplotlib.pyplot as plt

MINUTES_IN_DAY = 390
thresh = [20, 30, 40, 50, 60, 90]

x = np.linspace(0, MINUTES_IN_DAY, 1000)
for tr in thresh:
    weights_function = np.exp(-(MINUTES_IN_DAY - tr - x) / tr)
    plt.plot(x, weights_function, label=tr)

plt.legend()
plt.show()

fcn_weights = np.ones((1000,))

for tr in thresh:
    weights_function = np.exp(-(MINUTES_IN_DAY - tr - x) / tr)
    pr_fcn = fcn_weights / (fcn_weights + weights_function)
    plt.plot(x, pr_fcn, label=tr)

plt.legend()
plt.show()
