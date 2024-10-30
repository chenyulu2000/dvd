import numpy as np

data = np.loadtxt(open("visualization/learning_rate/learning_rate.csv", "rb"), delimiter=",", usecols=[0, 1])

iterations, lr = data[:, 0], data[:, 1]
print(iterations)
print(lr)

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Serif'
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(figsize=(10, 4), dpi=600)

ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(ls="--", lw=0.5, color="#4E616C")

ax.plot(iterations, lr, label='learning rate strategy')
plt.xlabel('iterations')
plt.ylabel('learning rate')
plt.legend(loc='best')

plt.savefig(f'visualization/learning_rate/learning_rate.pdf', dpi=600)
