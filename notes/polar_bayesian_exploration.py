import numpy as np
import matplotlib.pyplot as plt

radius = 1.5
theta = np.linspace(-np.pi, np.pi, 1000)
cos = radius * np.cos(theta)
sin = radius * np.sin(theta)
x = np.exp(cos) / (np.exp(cos) + np.exp(sin))
y = np.exp(sin) / (np.exp(cos) + np.exp(sin))

plt.plot(theta, x, label='x')
plt.plot(theta, y, label='y')
plt.legend()
plt.show()
