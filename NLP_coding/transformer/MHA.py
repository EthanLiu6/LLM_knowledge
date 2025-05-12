"""
MHA实现
"""
import torch
from torch import nn, Tensor


class MHA(nn.Module):
    def __init__(self, model_dim=1024, head_num=8, mask=None, dropout=0):
        super().__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        self.k_dim = self.model_dim // self.head_num

        self.q_k_v_c = nn.Linear(self.model_dim, self.model_dim * 4)

    def forward(self, in_x):
        seq_len = in_x.shape[1]

        q_k_v_c = self.q_k_v_c(in_x)
        query, key, value, short_cut = torch.split(q_k_v_c, self.model_dim, dim=-1)

        # multi-head and transpose
        query_head = query.reshape(-1, seq_len, self.head_num, self.k_dim).transpose(1, 2)
        key_head = key.reshape(-1, seq_len, self.head_num, self.k_dim).transpose(1, 2)
        value_head = value.reshape(-1, seq_len, self.head_num, self.k_dim).transpose(1, 2)

        # q @ k.T / sqrt(k_dim)
        atten_score = query_head @ key_head.transpose(-1, -2) / (self.k_dim ** 2)
        atten = torch.softmax(atten_score, dim=-1) @ value_head
        res = atten.transpose(1, 2).contiguous().view(in_x.shape)

        return res, atten_score, short_cut


class MQA(nn.Module):
    def __init__(self, model_dim, head_num):
        super(MQA, self).__init__()
        self.model_dim = model_dim
        self.head_num = head_num
        self.k_dim = self.model_dim // self.head_num

        self.short_cut = nn.Linear(model_dim, model_dim)
        # multi-head query, all query go to an only same key and value
        self.q_k_v = nn.Linear(self.model_dim, self.model_dim + self.k_dim * 2)

    def forward(self, in_x: Tensor):
        ipt_shape = in_x.shape  # (batch_size, seq_len, model_dim)

        short_cut = self.short_cut(in_x)

        # multi-head query, all query go to an only same key and value
        q_k_v = (self.q_k_v(in_x))
        query_head, same_one_key, same_one_value = torch.split(q_k_v, [self.model_dim, self.k_dim, self.k_dim], dim=-1)


        query_head = query_head.reshape(-1, ipt_shape[1], self.head_num, self.k_dim).transpose(1, 2)
        same_one_key = torch.unsqueeze(same_one_key, dim=2).transpose(1, 2)
        same_one_value = torch.unsqueeze(same_one_value, dim=2).transpose(1, 2)

        # MQA
        score = query_head @ same_one_key.transpose(-1, -2) / (self.k_dim ** 2)
        # (bs, head, seq, kd) @ (bs, 1, seq, kd) = (bs, head, seq, seq)
        atten = torch.softmax(score, dim=-1) @ same_one_value

        res = atten.transpose(1, 2).contiguous().view(ipt_shape)

        return score, atten, res, short_cut



if __name__ == '__main__':
    x = torch.randn(10, 32, 1024)
    mha = MHA(model_dim=1024, head_num=8)
    res, atten_score, short_cut = mha(x)
    print(res.shape)
    print(atten_score.shape)
    print(short_cut.shape)

    mqa = MQA(1024, 8)
    score, atten_res, res, short_cut = mqa(x)
    print(score.shape)
    print(atten_res.shape)
    print(res.shape)
    print(short_cut.shape)
