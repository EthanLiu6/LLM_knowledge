"""
矩阵乘法+矩阵梯度, 前面把权重的顺序调整的有点问题，和输出进行匹配了，不方便矩阵实现
这边直接和输入进行顺序匹配
"""
import numpy as np
import matplotlib.pyplot as plt


def w(i):
    return _w[i - 1]


def b(i):
    return _b[i - 1]


_w = np.linspace(start=0.1, stop=1.0, num=16)
w_layer1 = np.asarray([[w(1), w(2), w(3)],
                       [w(4), w(5), w(6)]])  # (2, 3)
w_layer2 = np.asarray([[w(7), w(8)],
                       [w(9), w(10)],
                       [w(11), w(12)]])  # (3, 2)
w_layer3 = np.asarray([[w(13), w(14)],
                       [w(15), w(16)]])  # (2, 2)
_b = np.asarray([0.025, 0.04, 0.03])
_xs = np.asarray([[3, 8],
                  [5, 10],
                  [16, 63]])
_ys = np.asarray([[-2, 5.25],
                  [-0.5, 7.2],
                  [3, 15]])
lr = 0.01
lr_drop_rate = 0.995

loss_record = []
epoch_num = 100


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def set_w_layer(w_layer: np.ndarray, grad: np.ndarray) -> np.ndarray:
    return w_layer - lr * grad


def net_work():
    """
    FP + BP
    """
    global w_layer3, w_layer2, w_layer1

    """FP: 前向传播， 损失计算"""
    # h11 = w(1) * x(1) + w(2) * x(2) + b(1)
    # h12 = w(3) * x(1) + w(4) * x(2) + b(1)
    # h13 = w(5) * x(1) + w(6) * x(2) + b(1)
    h_layer1 = _xs @ w_layer1 + b(1)

    # o11 = sigmoid(h11)
    # o12 = sigmoid(h12)
    # o13 = sigmoid(h13)
    o_layer1 = sigmoid(h_layer1)

    # h21 = w(7) * o11 + w(8) * o12 + w(9) * o13 + b(2)
    # h22 = w(10) * o11 + w(11) * o12 + w(12) * o13 + b(2)
    h_layer2 = o_layer1 @ w_layer2 + b(2)

    # o21 = sigmoid(h21)
    # o22 = sigmoid(h22)
    o_layer2 = sigmoid(h_layer2)

    # out31 = w(13) * o21 + w(14) * o22 + b(3)
    # out32 = w(15) * o21 + w(16) * o22 + b(3)
    out = o_layer2 @ w_layer3 + b(3)

    print("Pred:")
    # print(out31)
    # print(out32)
    print(out)
    print()

    # loss = 0.5 * (y(1) - out31) ** 2 + 0.5 * (y(2) - out32) ** 2
    loss = np.sum(0.5 * (_ys - out) ** 2)
    if loss_record and loss >= loss_record[-1]:
        return
    loss_record.append(loss)
    print("loss:", loss)

    """BP: 反向传播，求梯度 + 参数更新"""
    # NOTE: 对矩阵乘：X @ W = Y，L = f(Y), 有dL/dW = X^T @ dL/dY, dL/dX = dL/dY @ W^T

    # d_loss
    d_loss = out - _ys

    # layer3：w13, w14, w15, w16
    d_layer3_w = o_layer2.T @ d_loss  # (2, 2) @ (2, 2)

    # layer2: w7 - w12  # 这里还需要计算对o_layer2的偏导数
    d_o_layer2 = d_loss @ w_layer3.T  # (2, 2) @ (2, 2) = (2, 2)
    d_layer2_w = o_layer1.T @ (d_o_layer2 * (o_layer2 * (1 - o_layer2)))
    # (3, 2) @ [(2, 2) * (2, 2)] = (3, 2)

    # layer1: w1 - w6
    d_o_layer1 = (d_o_layer2 * o_layer2 * (1 - o_layer2)) @ w_layer2.T  # (2,2) @ (2,3) = (2,3)
    d_layer1_w = _xs.T @ (d_o_layer1 * o_layer1 * (1 - o_layer1))  # (2,2) @ (2,3) = (2,3)

    # update param, 从后向前
    # layer3
    w_layer3 = set_w_layer(w_layer3, d_layer3_w)
    w_layer2 = set_w_layer(w_layer2, d_layer2_w)
    w_layer1 = set_w_layer(w_layer1, d_layer1_w)


def show_loss():
    plt.plot(loss_record)
    plt.title('loss')
    plt.show()


if __name__ == '__main__':
    for epoch in range(epoch_num):
        # for i in range(len(_xs)):
        #     _x, _y = _xs[i], _ys[i]
        lr = lr * lr_drop_rate if lr > 1e-3 else lr
        net_work()
        print("epoch:", epoch)
    show_loss()

    print("w:", _w)
    print("lr:", lr)
