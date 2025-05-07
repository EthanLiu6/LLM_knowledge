"""
加载鸢尾花数据集并划分数据集
"""

import pandas as pd
from pandas import DataFrame
from config import ORIGIN_DATA_PATH, DATA_2_NUM_PATH
from sklearn.model_selection import train_test_split


def load_iris_data(
        data_path: str,
        _header=None,
        _names=None
) -> DataFrame:
    # if _names is None:
    #     _names = ['x1', 'x2', 'x3', 'x4', 'labels']

    return pd.read_csv(data_path, header=_header, names=_names)


def save_data_2_num_return(_data: DataFrame, save_path: str) -> DataFrame:
    # set labels to number class (object -> number)
    _data.iloc[:, -1], uniques = pd.factorize(_data.iloc[:, -1])
    # print(uniques)
    _data.to_csv(save_path)
    return _data


def train_test_data(_data: DataFrame,
                    _train_size,
                    _test_size):
    return train_test_split(_data, train_size=_train_size, test_size=_test_size, random_state=42)


if __name__ == '__main__':
    data = load_iris_data(ORIGIN_DATA_PATH)
    save_data_2_num_return(data, DATA_2_NUM_PATH)


    print(data.shape)
    # print(data.info())
    # print(train_test_data(data, 0.7, 0.3))
    train_data, test_data = train_test_data(data, 0.7, 0.3)
    print(test_data.shape)
    print(train_data.shape)
    print(test_data[:3])
