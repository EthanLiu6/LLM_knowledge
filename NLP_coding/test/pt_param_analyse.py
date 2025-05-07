import torch
import torch.nn as nn
from torch.nn.functional import softmax
# from torch.utils


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = nn.Linear(3, 512)
        self.l2 = nn.Linear(512, 2)

    def forward(self, ipt:torch.Tensor):
        h1 = softmax(self.l1(ipt), dim=-1)
        # o1 = nn.LayerNorm(h1)
        # h1.requires_grad = False
        return softmax(self.l2(h1), dim=-1)




if __name__ == '__main__':

    ipt = torch.rand(2, 4, 3)


    model = MyModel()
    # print(model._parameters)
    # print(model._buffers)
    # # print(model.modules)
    # print(model._modules)
    res = model(ipt)
    # print(list(model.buffers()))
    # print(len(list(model.parameters())))
    # print(list(model.parameters())[0].shape)
    # print(list(model.parameters())[1].shape)
    # print(list(model.parameters())[2].shape)
    # print(list(model.parameters())[3].shape)

    # print(ipt.shape)
    # print((torch.matmul(ipt, model.state_dict()['l1.weight'].T) + model.state_dict()['l1.bias']).shape)

    # print(model.state_dict()['l1.weight'].shape)
    # print(model.state_dict()['l1.bias'].shape)
    # print(model.state_dict()['l1.bias'].shape)
    # print(model.register_buffer)

    # print(list(model.named_buffers()))
    # print(list(model.named_children()))
    # print(list(model.children()))
    model.l1.weight.requires_grad = False
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"parameters: {params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
