import numpy as np
import pandas as pd


def read_iris():
    return pd.read_csv('./iris.csv', header=None, names=['x1', 'x2', 'x3', 'x4', 'labels'])




if __name__ == '__main__':
    # data = np.array(read_data())
    data = read_iris()
    # print(data.shape)
    # print(data.columns)

    # x = data.iloc[:, :-1]
    # x = data[['x1', 'x2', 'x3', 'x4']]
    # print(x)

    # labels = data.iloc[:, -1]
    # print(labels[:3])
    # print(labels.size)


    # print(data)
