import numpy as np


def naive(x, y):
    assert len(x.shape) == 2, 'x.shape 的长度不为2'
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]


x = np.random.random((2, 10, 2))
y = np.random.random((10, 2))
z = np.maximum(x, y)
# ad = np.add(x, y)
# print(x)
# print('*' * 50)
# print(y)
# print('*' * 50)
# print(ad)
# print('*' * 50)
