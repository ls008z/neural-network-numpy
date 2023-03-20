import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_gradient(x):
    return 1 - np.tanh(x)**2
