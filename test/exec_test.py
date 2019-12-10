import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('../')

from pysliceplorer import hyperslice
from pysliceplorer import sliceplorer

np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore', invalid='ignore')

def f(x, y, z):
    return z*((1 - np.sign(-x - .9 + abs(y * 2))) / 3 * (np.sign(.9 - x) + 1) / 3) * (np.sign(x + .65) + 1) / 2 - ((1 - np.sign(-x - .39 + abs(y * 2))) / 3 * (np.sign(.9 - x) + 1) / 3) + ((1 - np.sign(-x - .39 + abs(y * 2))) / 3 * (np.sign(.6 - x) + 1) / 3) * (np.sign(x - .35) + 1) / 2


def g(x, y):
    return np.sin(math.pi*x) / (math.pi*x) * np.sin(math.pi*y) / (math.pi*y)


dim = 3
c = (0, 0, 0.2)
test_var = hyperslice(f, -1, 1, dim, c, n_seg=100)

# plotting things so we can see
fig, axs = plt.subplots(dim, dim)

for i in range(0, dim):
    for j in range(0, dim):
        if i == 0:
            x_string = 'x' + str(j + 1)
            axs[i, j].set(xlabel=x_string)
            axs[i, j].xaxis.set_label_position('top')

        if j == 0:
            y_string = 'x' + str(dim - i)
            axs[i, j].set(ylabel=y_string)

        axs[i, j].pcolormesh(test_var.x_grid, test_var.y_grid,
                             test_var.data(j, dim - i - 1), cmap='pink')


test_var2 = sliceplorer(g, mn=-5, mx=5, dim=2, n_fpoint=160)
fig2, axs2 = plt.subplots(2)

# plot the 2nd figure
for i in range(0, test_var2.size):
    for j in range(0, test_var2.dim):
        axs2[j].plot(test_var2.x_grid, test_var2.data(i)['data'][j], color='k', alpha=0.1)


plt.show()