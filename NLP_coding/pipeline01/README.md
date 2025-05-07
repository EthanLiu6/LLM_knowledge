#                           01快速上手PyTorch



## 1. 能干嘛

- 把深度学习的模型结构和训练结构统一化

- 支持GPU上运算

- 支持onnx模型导出
- ……

## 2. 了解结构

[PyTorch](https://pytorch.org/)
[API Docs](https://pytorch.org/docs/stable/index.html)
[Torch.nn](https://pytorch.org/docs/stable/nn.html)
[pytorch仓库](https://github.com/pytorch/pytorch)

- 安装

- 数据结构：主要是`Tensor`（底层其实是numpy）

- 属性方法：……

- 几个重要模块：

    > Torch、torch.nn、torch.nn.functional、torch.Tensor、torch.optim、torch.utils.data

## 3. Init操作（简述）

- 初始化一些参数

- 初始化一些子模块或者所谓的layer（会注册到_modules属性）

    > 简单看看Linear的实现

- 初始化父类相关内容

    > ```python
    > """Initialize internal Module state, shared by both nn.Module and ScriptModule."""
    > ```



## 4. Module模块（简述）

- 所有模型结构继承自他
- 基本框架结构（官方demo）

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```

- Note：

    > As per the example above, an `__init__()` call to the parent class must be made before assignment on the child.

- callable

    > forward: Callable[..., Any] = _forward_unimplemented

- 几个重要属性

    > training、_parameters、_buffers、_modules等

- 几个重要的方法

    > state_dict、_load_from_state_dict、load_state_dict、parameters、named_parameters、buffers、named_buffers、children、named_children、modules、named_modules、train、eval、requires_grad_、zero_grad、to
    >
    > 有很多注册信息，其实是弄到state_dict等内容



## 5. Pipline 

回想一下，有哪些流程？

![image-20250414213944819](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250414213944819.png)

### 5.1 模型初步搭建（BP结构）

> 三层全连接+2层激活
>
> 随机初始化数据跑通

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, matmul, Tensor


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


```



### 5.2 数据处理与加载

> 先不讲解dataloader，先使用机器学习的方法构建
>
> 拿鸢尾花数据集来操作

```python
"""
加载鸢尾花数据集并实现分类任务训练
"""

import pandas as pd
from pandas import DataFrame
from config import DATA_PATH
from sklearn.model_selection import train_test_split


def load_iris_data(
        data_path: str,
        _header=None,
        _names=None
) -> DataFrame:
    
    if _names is None:
        _names = ['x1', 'x2', 'x3', 'x4', 'labels']
        
    return pd.read_csv(data_path, header=_header, names=_names)


def train_test_data(_data: DataFrame, 
                    _train_size, 
                    _test_size):
    
    return train_test_split(_data, train_size=_train_size, test_size=_test_size, random_state=42)



if __name__ == '__main__':
    data = load_iris_data(DATA_PATH)
    print(data.shape)
    # print(data.info())
    # print(train_test_data(data, 0.7, 0.3))
    train_data, test_data = train_test_data(data, 0.7, 0.3)
    print(test_data.shape)
    print(train_data.shape)
    print(test_data[:3])

```

### 5.3 baseline跑通

> 模型训练与预测

```python

```



### 5.4 评估指标

- 不同类型的模型使用不同的评估指标
- 不同的业务场景，对评估指标的选择偏好不同

#### 问勒问题

![image-20250414214456421](https://coderethan-1327000741.cos.ap-chengdu.myqcloud.com/blog-pics/image-20250414214456421.png)

模型的测试一般以下几个方面来进行比较，在分类算法中常见的指标分别是准确率/召回率/精准率/F值(F1指标)

• 准确率(Accuracy)=提取出的正确样本数/总样本数

• 召回率(Recall)=正确的正例样本数/样本中的正例样本数——覆盖率

• 精准率(Precision)=正确的正例样本数/预测为正例的样本数

• F值=Precision✖️Recall✖️2 / (Precision+Recall) (即F值为精准率和召回率的调和平均值)

• ROC曲线

• AUC值

#### 回归问题

• 绝对值误差

•  均方误差

• $R^2$

### 5.4 模型持久化



## 6. Others

- 没事干就多转转官网
- 一定一定要自己手敲一边！！！

### 7. 讲了些啥？

> 问问自己，你学到了啥？可否使用起来？