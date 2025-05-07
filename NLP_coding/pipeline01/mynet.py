import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, matmul, Tensor

__all__ = ['MyNet']


class MyNet(nn.Module):
    r"""MyNet conclude 3 FN and 2 activation which used ReLU and sigmoid.

    Args:
        in_features: size of each input sample(feature).
        out_features: size of each output sample(feature). Or maybe you need size of predicted labels.

    forward():
        must be impl, which input is 'input features'. shape like (N, in_features)
    """

    # init some model 'input param', submodule etc.
    def __init__(
            self,
            in_features: int,
            out_features: int
    ) -> None:
        super(MyNet, self).__init__()  # init Base Module message.

        #  Parameter can default set `requires_grad=True`
        self.layer1_weight = nn.Parameter(torch.empty(in_features, 12))  # (in_features, 12)
        self.layer1_bias = nn.Parameter(torch.empty(12))
        self.layer2_weight = nn.Parameter(torch.empty(12, 8))  # (12, 8)
        self.layer2_bias = nn.Parameter(torch.empty(8))
        self.layer3_weight = nn.Parameter(torch.empty(8, out_features))  # (8, out_features)
        self.layer3_bias = nn.Parameter(torch.empty(out_features))
        # 3 nn.Linear()

        # print(self.layer1_bias)
        self._reset_param()
        # print(self.layer1_bias)

    def _reset_param(self):
        for param in self.parameters():
            nn.init.normal_(param)

    def forward(self, ipt: Tensor) -> Tensor:
        h1 = add(matmul(ipt, self.layer1_weight), self.layer1_bias)
        o1 = F.relu(h1)
        h2 = add(matmul(o1, self.layer2_weight), self.layer2_bias)
        o2 = F.sigmoid(h2)
        o3 = add(matmul(o2, self.layer3_weight), self.layer3_bias)

        return o3


if __name__ == '__main__':
    ipt = torch.randn(4, 6)
    my_net = MyNet(6, 3)
    out = my_net(ipt)  # callable
    print(out.shape)
