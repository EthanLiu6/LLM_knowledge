"""
 DyT论文部分解读
"""
import numpy as np

from activations import tanh, d_tanh
from activations import show, show_single


def DyT(_x: np.array, _alpha: float, _gamma: float = 1.0, _beta: float = 0):
    """
    out = \gamma * Tanh(\alpha * x) + \beta
    """
    return _gamma * tanh(alpha * _x) + _beta


def d_DyT(_x: np.array, _alpha: float, _gamma: float = 1.0, _beta: float = 0):
    return d_tanh(_x * _alpha)


def norm(_x: np.array, _gama: float = 1.0, _beta: float = 0.0):
    """
    out = \gamma * \frac{(_x - E_{_x})}{\sqrt (Var_{_x})} + \beta
    """
    eps = 1e-5
    mean = np.mean(_x)
    std = np.std(_x)
    return _gama * (_x - mean) / (std + eps) + _beta


def d_norm():
    pass


if __name__ == '__main__':
    n = 10
    step = 0.2
    alpha = 0.5
    # alpha = 1/4
    # alpha = 1
    x = np.arange(-n, n, step)
    y = DyT(x, alpha)
    dy = d_DyT(y, alpha)
    show(_x=x, _y=y, _dy=dy, _title='DyT')

    # x = x
    # y = norm(_x=x)
    # show_single(_x=x, _y=y, _title='Norm')


