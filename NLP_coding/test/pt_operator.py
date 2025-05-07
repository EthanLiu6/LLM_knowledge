"""
02-PyTorch的一些常用算子
"""
import torch
import torch.nn as nn
from torch import Tensor


def t1(ipt: Tensor, out_dim):
    l1 = nn.Linear(ipt.shape[-1], out_dim)
    print(l1.weight.size())
    return l1(ipt)


def t2():
    # With Learnable Parameters
    m = nn.BatchNorm1d(100)
    # Without Learnable Parameters
    m = nn.BatchNorm1d(100, affine=False)
    input = torch.randn(20, 100)
    output = m(input)
    print(output.shape)

    # With Learnable Parameters
    # m = nn.BatchNorm2d(100)
    # # Without Learnable Parameters
    # m = nn.BatchNorm2d(100, affine=False)
    # input = torch.randn(20, 100, 35, 45)
    # output = m(input)
    # rms_norm = nn.RMSNorm([2, 3])
    # input = torch.randn(2, 2, 3)
    # output = rms_norm(input)
    # print(output)


def t3():
    x = torch.randn(2, 3, 5)
    new = torch.permute(x, (2, 0, 1))
    new = torch.reshape(x, (5, 3, 2))
    new = torch.transpose(x, 1, 2)
    new = x.view(1, 3, 10)
    print(x.size())
    print(new.size())
    print(x[0][0][0] == new[0][0][0])
    print(id(x[0][0][0]) == id(new[0][0][0]))
    print(x.data_ptr())
    print(new.data_ptr())
    print(x.stride())
    print(new.stride())
    print(x.is_contiguous())
    print(new.is_contiguous())
    x = x.reshape(5, 2, 3)
    new = x.view(1, 3, 10)
    print(x.is_contiguous())
    print(new.is_contiguous())
    # x = x.transpose(1, 2)
    # print(x.is_contiguous())
    # new = x.view(1, 3, 10)  # maybe error
    # print(new.is_contiguous())
    x = x.permute([0, 2, 1])
    print(x.is_contiguous())
    # new = x.view(1, 3, 10)  # maybe error


def t4():
    # embeddinmg

    # an Embedding module containing 10 tensors of size 3
    embedding = nn.Embedding(9, 3)
    print(embedding.weight.size())
    # a batch of 2 samples of 4 indices each
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 8]])
    print(embedding(input))

    # example with padding_idx
    embedding = nn.Embedding(10, 3, padding_idx=0)
    input = torch.LongTensor([[0, 2, 0, 5]])
    embedding(input)

    # example of changing `pad` vector
    padding_idx = 0
    embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
    print(embedding.weight)
    with torch.no_grad():
        embedding.weight[padding_idx] = torch.ones(3)
    print(embedding.weight)

    # FloatTensor containing pretrained weights
    weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
    embedding = nn.Embedding.from_pretrained(weight)
    # Get embeddings for index 1
    input = torch.LongTensor([1])
    print(embedding(input))


def t5():
    m = nn.Dropout(p=0.2)
    input = torch.randn(20, 16)
    output = m(input)
    print(output.shape)


if __name__ == '__main__':
    # t01 = torch.randn(2, 3)
    # print(t1(t01, 10).shape)

    # debug check weight

    # # print(t01.data)
    # print(t01.storage())
    # print(t01.data_ptr())
    # print(t01.data_ptr())
    # print(t01.stride())
    # print(t01.contiguous())
    # print(torch.storage)

    # t2()

    # t3()

    # t4()

    t5()