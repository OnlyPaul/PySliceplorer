import math
import numpy as np
import pysliceplorer as psp


np.set_printoptions(threshold=np.nan)
np.seterr(divide='ignore', invalid='ignore')


def f(x, y, z):
    return z*((1 - np.sign(-x - .9 + abs(y * 2))) / 3 * (np.sign(.9 - x) + 1) / 3) \
           * (np.sign(x + .65) + 1) / 2 - ((1 - np.sign(-x - .39 + abs(y * 2))) / 3 * (np.sign(.9 - x) + 1) / 3) \
           + ((1 - np.sign(-x - .39 + abs(y * 2))) / 3 * (np.sign(.6 - x) + 1) / 3) * (np.sign(x - .35) + 1) / 2


def g(s, t, u, v):
    return np.sin(math.pi*s) / (math.pi*s) * np.sin(math.pi*t) / (math.pi*t) * np.sin(math.pi*u) / (math.pi*u) * np.sin(math.pi*v) / (math.pi*v)


def h(s, t, u, v):
    return ((s**4 - 16*s**2 + 5*s)+(t**4 - 16*t**2 + 5*t)+(u**4 - 16*u**2 + 5*u)+(v**4 - 16*v**2 + 5*v))/2


dim = 4
mn = -5
mx = 5

psp.sliceplorer(f, mn=-1.5, mx=1.5, dim=3, n_fpoint=100, height=225, width=450, output='vis_spl_test_A.html')

psp.hyperslice(h, mn=mn, mx=mx, dim=4, fpoint=(0, 0, 0, 0), n_seg=100)

"""
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
"""

vertices = [
    [-1, -1, -1],
    [1, -1, 1],
    [-1, 1, 1],
    [1, 1, -1]
]

config = [
    [0, 1, 2],
    [0, 3, 1],
    [3, 2, 1],
    [0, 3, 2]
]

mn = -1.2
mx = 1.2

psp.hypersliceplorer(vertices=vertices, config=config, mn=mn, mx=mx, n_fpoint=15,
                     output='vis_hsp_test.html', width=400, height=400)

