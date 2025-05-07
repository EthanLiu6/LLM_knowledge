"""
多样本
"""
import numpy as np
import matplotlib.pyplot as plt

_w = np.linspace(start=0.1, stop=1.0, num=16)
_b = np.asarray([0.025, 0.04, 0.03])
_xs = np.asarray([[3, 8],
                 [5, 2]])
_ys = np.asarray([[-0.35, 1.25],
                 [0.8, 0.45]])
lr = 0.03
lr_drop_rate = 0.99

loss_record = []
epoch_num = 150


def w(i):
    return _w[i - 1]


def b(i):
    return _b[i - 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def x(i):
    return _x[i - 1]


def y(i):
    return _y[i - 1]


def set_w(i, grad):
    i = i - 1
    _w[i] = _w[i] - lr * grad


def net_work():
    """
    FP + BP
    """

    """FP: 前向传播， 损失计算"""
    h11 = w(1) * x(1) + w(2) * x(2) + b(1)
    h12 = w(3) * x(1) + w(4) * x(2) + b(1)
    h13 = w(5) * x(1) + w(6) * x(2) + b(1)

    o11 = sigmoid(h11)
    o12 = sigmoid(h12)
    o13 = sigmoid(h13)

    h21 = w(7) * o11 + w(8) * o12 + w(9) * o13 + b(2)
    h22 = w(10) * o11 + w(11) * o12 + w(12) * o13 + b(2)

    o21 = sigmoid(h21)
    o22 = sigmoid(h22)

    out31 = w(13) * o21 + w(14) * o22 + b(3)
    out32 = w(15) * o21 + w(16) * o22 + b(3)

    print("Pred:")
    print(out31)
    print(out32)
    print()

    loss = 0.5 * (y(1) - out31) ** 2 + 0.5 * (y(2) - out32) ** 2
    if loss_record and loss > loss_record[-1]:
        return
    loss_record.append(loss)
    print("loss:", loss)

    """BP: 反向传播，求梯度 + 参数更新"""
    # d_loss
    d_loss1 = (out31 - y(1))
    d_loss2 = (out32 - y(2))

    # layer3：w13, w14, w15, w16
    d_out31_w13 = o21
    d_out31_w14 = o22
    d_out32_w15 = o21
    d_out32_w16 = o22

    # layer2: w7 - w12
    d_o21 = o21 * (1 - o21)
    d_o22 = o22 * (1 - o22)

    d_h21_w7 = o11
    d_h21_w8 = o12
    d_h21_w9 = o13
    d_h22_w10 = o11
    d_h22_w11 = o12
    d_h22_w12 = o13

    # layer1: w1 - w6
    d_o11 = o11 * (1 - o11)
    d_o12 = o12 * (1 - o12)
    d_o13 = o13 * (1 - o13)

    d_h11_w1 = x(1)
    d_h11_w2 = x(2)
    d_h12_w3 = x(1)
    d_h12_w4 = x(2)
    d_h13_w5 = x(1)
    d_h13_w6 = x(2)

    # update param, 从后向前
    # layer3
    set_w(13, grad=d_loss1 * d_out31_w13)
    set_w(14, grad=d_loss1 * d_out31_w14)
    set_w(15, grad=d_loss2 * d_out32_w15)
    set_w(16, grad=d_loss2 * d_out32_w16)

    # layer2 (include d_sigmoid)
    # 计算从输出层传递到第二层的梯度
    grad_o21 = (d_loss1 * w(13) + d_loss2 * w(15)) * d_o21
    grad_o22 = (d_loss1 * w(14) + d_loss2 * w(16)) * d_o22

    set_w(7, grad=grad_o21 * d_h21_w7)
    set_w(8, grad=grad_o21 * d_h21_w8)
    set_w(9, grad=grad_o21 * d_h21_w9)
    set_w(10, grad=grad_o22 * d_h22_w10)
    set_w(11, grad=grad_o22 * d_h22_w11)
    set_w(12, grad=grad_o22 * d_h22_w12)

    # layer1 (include d_sigmoid)
    # 计算从第二层传递到第一层的梯度
    grad_o11 = (grad_o21 * w(7) + grad_o22 * w(10)) * d_o11
    grad_o12 = (grad_o21 * w(8) + grad_o22 * w(11)) * d_o12
    grad_o13 = (grad_o21 * w(9) + grad_o22 * w(12)) * d_o13

    set_w(1, grad=grad_o11 * d_h11_w1)
    set_w(2, grad=grad_o11 * d_h11_w2)
    set_w(3, grad=grad_o12 * d_h12_w3)
    set_w(4, grad=grad_o12 * d_h12_w4)
    set_w(5, grad=grad_o13 * d_h13_w5)
    set_w(6, grad=grad_o13 * d_h13_w6)


def show_loss():
    plt.plot(loss_record)
    plt.title('loss')
    plt.show()


if __name__ == '__main__':
    for epoch in range(epoch_num):
        for i in range(len(_xs)):
            _x, _y = _xs[i], _ys[i]
        lr = lr * lr_drop_rate if lr > 1e-3 else lr
        net_work()
        print("epoch:", epoch)
    show_loss()

    print("w:", _w)
    print("lr:", lr)
