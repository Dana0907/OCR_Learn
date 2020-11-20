import numpy as np


# 两个向量之间的点积是一个标量
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# 矩阵与向量的点积是一个向量
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z


# 两个矩阵之间的点积
# 当且仅当 x.shape[1] == y.shape[0] 时，你才可以对它们做点积（dot(x, y)）
# 得到的结果是一个形状为 (x.shape[0], y.shape[1]) 的矩阵
# 其元素为 x的行与 y 的列之间的点积
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


if __name__ == '__main__':
    # x = np.array([1, 2, 3])
    # y = np.array([1, 2, 3])
    #
    # print(naive_vector_dot(x, y))
    # print(np.dot(x, y))
    #
    # x = np.array([[1, 2, 3], [1, 2, 3]])
    # y = np.array([1, 2, 3])
    #
    # print(naive_matrix_vector_dot(x, y))
    # print(np.dot(x, y))

    x = np.array([[1, 2, 3], [1, 2, 3]])
    y = np.array([[1, 2, 3], [1, 2, 3]])
    # y = np.transpose(y)
    # print(y)
    y = y.reshape(3, 2)
    print(y)
    print(naive_matrix_dot(x, y))
    print(np.dot(x, y))
