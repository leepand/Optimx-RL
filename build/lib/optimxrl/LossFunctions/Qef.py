import numpy as np


class Qef:
    """
    quadratic error function ||prediction-target||²
    """

    def __init__(self):
        pass

    def Loss(self, ya, yta, dev=False):
        if dev == True:
            return yta - ya
        return np.sum((yta - ya) ** 2) / (yta.shape[0] * 2.0)
