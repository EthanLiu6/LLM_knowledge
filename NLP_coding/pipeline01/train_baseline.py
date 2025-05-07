import os
import sys

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
import numpy as np
import matplotlib.pyplot as plt
from mynet import MyNet
from my_metrics import accuracy
from iris_data_process import load_iris_data, save_data_2_num_return, train_test_data
from config import ORIGIN_DATA_PATH, DATA_2_NUM_PATH

import json

_config: dict

_config = {
    '_origin_data': load_iris_data(data_path=ORIGIN_DATA_PATH),
    '_lr': 0.01,
    '_losses': [],
    '_batch_size': 6,
    '_epoch_num': 300,
    'save_model_dir': r'./save_model/'
}

model = MyNet(in_features=4, out_features=3)
# optim = SGD(model.parameters(), lr=_config['_lr'])
optim = Adam(model.parameters(), lr=_config['_lr'])
loss_func = nn.CrossEntropyLoss()


class TrainAndPred:
    def __init__(self):
        self.test_x = None
        self.test_y = None

    # need: train and pred data, model, lr, batch_size, epoch, loss, etc.
    def train(self, config_dict):
        model.train()
        for epoch in range(config_dict['_epoch_num']):
            # 1. data module
            _origin_data = config_dict['_origin_data']
            data = save_data_2_num_return(_origin_data, DATA_2_NUM_PATH)

            # this date type is DataFrame
            train_data, test_data = train_test_data(data, _train_size=0.7, _test_size=0.3)
            train_x, train_y, test_x, test_y = \
                train_data.iloc[:, :-1], train_data.iloc[:, -1], test_data.iloc[:, :-1], test_data.iloc[:, -1]

            train_x = np.asarray(train_x)
            train_y = np.asarray(train_y).astype(np.float64)  # need object to number type, 计算损失的时候还要用Long

            test_x = np.asarray(test_x)
            test_y = np.asarray(test_y).astype(np.float64)

            self.test_x = test_x
            self.test_y = test_y

            # 2. forward module

            optim.zero_grad()  # need grad zero
            # Note: input must be Tensor
            tensor_train_x = torch.tensor(train_x, dtype=torch.float32)
            train_out = model(tensor_train_x)

            # 3. loss calculate and backward, and model save
            # Note: need tensor 拖 numpy，and pred need label class
            # TODO Bug: pred_y = np.argmax(train_out.detach().numpy(), axis=1)
            # already repaired: 交叉熵使用的时候不需要获取对应预测类别，这点我是真不了解，后续补补交叉熵的代码逻辑

            if epoch % 5 == 0 or epoch == config_dict['_epoch_num'] - 1:
                pred_y = np.argmax(train_out.detach().numpy(), axis=1)
                print("pred y：", pred_y)
                print("Accuracy：", accuracy(y_true=train_y, y_pred=pred_y))

                if not os.path.exists(config_dict['save_model_dir']):
                    os.makedirs(config_dict['save_model_dir'])
                torch.save(model, config_dict['save_model_dir'] + f'{epoch}_model.pth')
            # TODO bug: loss = loss_func(torch.tensor(pred_y, dtype=torch.float32, requires_grad=True),
            #                  torch.tensor(train_y, dtype=torch.float32, requires_grad=True))

            loss = loss_func(train_out, torch.tensor(train_y, dtype=torch.long))
            config_dict['_losses'].append(loss.item())
            print(loss)

            if epoch == config_dict['_epoch_num'] - 1:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'epoch': epoch,
                    'loss': loss,
                }, config_dict['save_model_dir'] + 'checkpoint.pth')

            loss.backward()

            # 4. update param
            optim.step()

    @torch.no_grad()
    def test(self, _model):
        print('=' * 30)
        print('Test:')
        test_model = torch.load(_model, weights_only=False)
        test_x = self.test_x
        test_y = self.test_y
        tensor_test_x = torch.tensor(test_x, dtype=torch.float32)
        test_out = test_model(tensor_test_x)
        pred_y = np.argmax(test_out.detach().numpy(), axis=1)
        print("Acc:", accuracy(test_y, pred_y))


def show_loss():
    plt.plot(_config['_losses'])
    plt.title('loss')
    plt.show()


if __name__ == '__main__':
    run = TrainAndPred()
    run.train(_config)
    show_loss()

    run.test(r'./save_model/299_model.pth')


