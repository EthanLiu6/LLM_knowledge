"""
适用于分类和回归的评估指标
"""
import numpy as np
from numpy import ndarray

__all__ = ['accuracy', 'precision', 'recall', 'f1', 'auc',
           'MAE', 'MSE', 'R2']


def accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """ pred right / all"""
    return sum(y_true == y_pred) / len(y_true)


def precision():
    """need OVR setting for multi-class"""
    pass


def recall():
    pass


def f1():
    pass


def auc():
    # TODO: will add in other days
    pass


def MAE(y_true: ndarray[float], y_pred: ndarray[float]) -> ndarray:
    return np.sum(np.abs(y_true - y_pred) / len(y_true))


def MSE(y_true: ndarray[float], y_pred: ndarray[float]) -> ndarray:
    return np.sum(np.power(np.abs(y_true - y_pred), 2) / len(y_true))


def R2():
    pass


if __name__ == '__main__':
    y1 = np.asarray([3, 3, 2.5, 1])
    y2 = np.asarray([1, 1, 4, 1])
    print(accuracy(y1, y2))
    print(MAE(y1, y2))
    print(MSE(y1, y2))
