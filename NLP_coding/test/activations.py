"""
01-激活函数相关内容
"""

import numpy as np
import matplotlib.pylab as plt


def sigmoid(_x: np.array):
    return 1 / (1 + np.exp(-_x))


def d_sigmoid(_x: np.array):
    return _x * (1 - _x)


def tanh(_x: np.array):
    return (np.exp(_x) - np.exp(-_x)) / (np.exp(_x) + np.exp(-_x))


def d_tanh(_x: np.array):
    return 1 - pow(_x, 2)


def relu(_x: np.array):
    return np.array([x if x > 0 else 0 for x in _x])


def d_relu(_x: np.array):
    return np.array([1 if x > 0 else 0 for x in _x])


def relu6(_x: np.array):
    pass


def d_relu6(_x: np.array):
    pass


def elu(_x: np.array, a: float = 0.5):
    """
    if x > 0, res = x; else, res = a(np.pow(np.e, x) - 1)
    """
    return [x if x > 0 else (a * (np.exp(x) - 1)) for x in _x]


def d_elu(_x: np.array, a: float = 0.5):
    return np.array([1 if x > 0 else (a * np.exp(x)) for x in _x])


def gelu(_x: np.array):
    return 0.5 * _x * (1 + tanh(np.sqrt(2 / np.pi) * (_x + 0.044715 * _x ** 3)))


def d_gelu(_x: np.array):
    # 计算 tanh 部分的导数
    u = np.sqrt(2 / np.pi) * (_x + 0.044715 * _x ** 3)
    tanh_u = np.tanh(u)
    d_tanh_u = 1 - tanh_u ** 2

    # GELU 的导数
    return 0.5 * (1 + tanh_u) + 0.5 * _x * d_tanh_u * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * _x ** 2)


def softmax(_x: np.array):
    # denom = np.sum(pow(np.e, m) for m in _x)  # the same data （one data）has same denom
    # return np.array([x / denom for x in _x])

    # output = np.zeros_like(_x, dtype=np.float64)
    # max_xi = np.max(_x)
    # exps = np.exp(_x - max_xi)
    # sum_xj = np.sum(exps)
    # for i in range(len(_x)):
    #     output[i] = exps[i] / sum_xj
    #
    # return output

    # another easier impl
    exps = np.exp(_x - np.max(_x))
    sum_xj = np.sum(exps)
    return exps / sum_xj


def d_softmax(_x: np.array):
    """
    计算softmax的雅可比矩阵（偏导数矩阵）
    """
    # 计算softmax的输出

    # 初始化雅可比矩阵
    jacobian = np.zeros((len(_x), len(_x)), dtype=np.float64)

    # 填充雅可比矩阵
    for i in range(n):
        for j in range(n):
            if i == j:
                # 对角线元素：y_i * (1 - y_i)
                jacobian[i, j] = _x[i] * (1 - _x[i])
            else:
                # 非对角线元素：-y_i * y_j
                jacobian[i, j] = -_x[i] * _x[j]

    return jacobian


"""show func"""


def show(_x, _y, _dy, _title=''):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(_x, _y, 'r')
    axs[0].set_title(_title)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    axs[1].plot(_x, _dy, 'b')
    axs[1].set_title('d_' + _title)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('d y')

    plt.tight_layout()
    plt.show()


def show_compare(_x, _y1, _dy1, _y2, _dy2, label1='Func1', label2='Func2', _title='Comparison'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # 子图 1: 函数值对比
    axs[0].plot(_x, y1, 'r', label=label1)
    axs[0].plot(_x, y2, 'g', label=label2)
    axs[0].set_title(_title)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].legend()
    axs[0].grid(True)

    # 子图 2: 导数对比
    axs[1].plot(_x, dy1, 'r', label='d_' + label1)
    axs[1].plot(_x, dy2, 'g', label='d_' + label2)
    axs[1].set_title('Derivatives of ' + _title)
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('dy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def show_single(_x, _y, _title):
    plt.plot(_x, _y, 'b')
    plt.title(_title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    n = 3
    x = np.arange(-n, n, 0.2)

    # y = sigmoid(x)
    # dy = d_sigmoid(y)
    # show(x, y, dy, 'sigmoid')

    # y = tanh(x)
    # dy = d_tanh(y)
    # show(x, y, dy, 'tanh')

    # y = relu(x)
    # dy = d_relu(y)
    # show(x, y, dy, 'relu')

    # y1 = sigmoid(x)
    # dy1 = d_sigmoid(y1)
    # y2 = tanh(x)
    # dy2 = d_tanh(y2)
    # show_compare(_x=x, _y1=y1, _dy1=dy1, _y2=y2, _dy2=dy2, label1='sigmoid', label2='tanh')

    # y = elu(x, a=1)
    # dy = d_elu(y, a=1)
    # show(x, y, dy, 'elu')

    # y = gelu(x)
    # dy = d_gelu(y)
    # show(x, y, dy, 'gelu')

    y1 = gelu(x)
    dy1 = d_gelu(y1)
    y2 = relu(x)
    dy2 = d_relu(y2)
    show_compare(_x=x, _y1=y1, _dy1=dy1, _y2=y2, _dy2=dy2, label1='gelu', label2='relu')

    # y = softmax(x)
    # dy = d_softmax(y)
    # print(dy.shape)
