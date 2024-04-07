"""
Created by : Leepand
General utility functions.
"""

import numpy as np

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.testing import assert_allclose


def safe_div(a, b):
    """Returns a if b is nil, else divides a by b.
    When scaling, sometimes a denominator might be nil. For instance, during standard scaling
    the denominator can be nil if a feature has no variance.
    """
    return a / b if b else 0.0


def get_random_normalized_vector(dim: int) -> np.array:
    """
    Returns a random normalized vector with the given dimensions.

    Args:
        dim (int): The dimensionality of the output vector.

    Returns:
        numpy.array: A random normalized vector that lies on the surface of the :py:attr:`dim`-dimensional hypersphere.
    """
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


class MemoryCache:
    def __init__(self, n):
        self.cache = {}
        self.capacity = n

    def __call__(self, shape, dtype=np.float64):
        key = (shape, dtype)
        if key in self.cache:
            return self.cache.pop(key)
        return np.zeros(shape, dtype=dtype)

    def __setitem__(self, key, value):
        self.cache[key] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def test_MemoryCache():
    n = 2
    c = MemoryCache(n)

    # Concatenate 100 tensors of size (3, 3) along dimension 0
    x = c((n, 3, 3))
    for i in range(n):
        x[i] = np.random.random((3, 3))


def initialize_parameters(n_in, n_out, ini_type="plain", opt="normal"):
    """
    Helper function to initialize some form of random weights and Zero biases
    Args:
        n_in: size of input layer
        n_out: size of output/number of neurons
        ini_type: set initialization type for weights

    Returns:
        params: a dictionary containing W and b
    """

    params = dict()  # initialize empty dictionary of neural net parameters W and b

    if ini_type == "plain":
        params["W"] = (
            np.random.randn(n_out, n_in) * 0.01
        )  # set weights 'W' to small random gaussian
    elif ini_type == "xavier":
        params["W"] = np.random.randn(n_out, n_in) / (
            np.sqrt(n_in)
        )  # set variance of W to 1/n
    elif ini_type == "he":
        # Good when ReLU used in hidden layers
        # Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Kaiming He et al. (https://arxiv.org/abs/1502.01852)
        # http: // cs231n.github.io / neural - networks - 2 /  # init
        params["W"] = np.random.randn(n_out, n_in) * np.sqrt(
            2 / n_in
        )  # set variance of W to 2/n

    params["b"] = np.zeros((n_out, 1))  # set bias 'b' to zeros
    if opt in ["rmsprop", "adam"]:
        # _n_in_w,_n_out_w =params["W"].shape
        _n_in_b, _n_out_b = params["b"].shape
        params["m1"] = {
            "W": np.random.uniform(0.1, 1, params["W"].shape),
            "b": np.random.uniform(0.1, 1, params["b"].shape),
            # "b": np.zeros((_n_out_b, 1)),
        }
        params["m2"] = {
            "W": np.random.uniform(0.1, 1, params["W"].shape),
            "b": np.random.uniform(0.1, 1, params["b"].shape),
            # "b": np.zeros((_n_out_b, 1)),
        }
    return params
