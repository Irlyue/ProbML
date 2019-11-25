import numpy as np


def softmax(x):
    """
    :param x: np.array, shape(m, n)
    :return: np.array, shape(m, n)
    """
    x = x - x.max(axis=1, keepdims=True)
    expX = np.exp(x)
    probs = expX / expX.sum(axis=1, keepdims=True)
    return probs