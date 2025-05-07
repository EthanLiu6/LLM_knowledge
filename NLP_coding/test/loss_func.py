"""
一些损失函数相关内容
"""
import torch
import torch.nn as nn


def t1():
    # mse
    loss = nn.L1Loss()
    y1 = torch.randn(3, 5, requires_grad=True)
    y2 = torch.randn(3, 5)
    for i in range(10):
        out = loss(y1, y2)
        print(out)
        out.backward()


def t2():
    # l2
    loss = nn.MSELoss()
    y1 = torch.randn(3, 5, requires_grad=True)
    print(y1)
    y2 = torch.randn(3, 5)
    for i in range(10):
        out = loss(y1, y2)
        print(out)
        out.backward()


def t3():
    # CE
    loss = nn.CrossEntropyLoss()
    y1 = torch.randn(3, 5, requires_grad=True)
    print(y1)
    y2 = torch.randn(3, 5)
    for i in range(10):
        out = loss(y1, y2)
        print(out)
        out.backward()


if __name__ == '__main__':
    t3()
