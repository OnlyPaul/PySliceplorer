import math
import numpy as np
import sys

sys.path.append('../')
import pysliceplorer as psp


np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore', invalid='ignore')


def f(x, y, z):
    return z*((1 - np.sign(-x - .9 + abs(y * 2))) / 3 * (np.sign(.9 - x) + 1) / 3) \
           * (np.sign(x + .65) + 1) / 2 - ((1 - np.sign(-x - .39 + abs(y * 2))) / 3 * (np.sign(.9 - x) + 1) / 3) \
           + ((1 - np.sign(-x - .39 + abs(y * 2))) / 3 * (np.sign(.6 - x) + 1) / 3) * (np.sign(x - .35) + 1) / 2


def g(x, y):
    return np.sin(math.pi*x) / (math.pi*x) * np.sin(math.pi*y) / (math.pi*y)


dim = 2
mn = -5
mx = 5

psp.sliceplorer(g, mn=mn, mx=mx, dim=dim, n_fpoint=100, height=400, width=800, output='vis_spl_test.html')

psp.hyperslice(f, mn=-1, mx=1, dim=3, fpoint=(0, 0, 0.2), n_seg=100)

vertices = [
    [-1, 1, 1],
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, 1],
    [-1, 1, -1],
    [1, 1, -1],
    [-1, -1, -1],
    [1, -1, -1]
]
config = [
    [1, 0, 2], # top face
    [3, 1, 2],
    [6, 4, 7], # bottom face
    [7, 5, 4],
    [6, 2, 4], # left face
    [2, 4, 0],
    [7, 3, 2], # front face
    [6, 7, 2],
    [7, 5, 1], # right face
    [7, 1, 3],
    [5, 4, 1], # back face
    [1, 4, 0]
]

mn = -1.2
mx = 1.2

psp.hypersliceplorer(vertices=vertices, config=config, mn=mn, mx=mx, n_fpoint=15,
                     output='vis_hsp_test.html', width=400, height=400)

