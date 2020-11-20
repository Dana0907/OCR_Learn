import numpy as np

# 标量  也叫标量张量、零维张量、0D 张量
i = np.array(12)

# 向量 1D张量
x = np.array([1, 2, 4, 6])
x1 = np.array([6, 2, 5, 5])

# 矩阵  2D张量
y = np.array([[1, 2, 4], [2, 44, 5]])

# 3D张量
z = np.array([[[1, 2, 4], [2, 44, 5]], [[1, 2, 4], [2, 44, 5]], [[1, 2, 4], [2, 44, 5]]])

# 将多个 3D 张量组合成一个数组，可以创建一个 4D 张量，以此类推。深度学习处理的一般是 0D 到 4D 的张量，但处理视频数据时可能会遇到 5D 张量

print(x)
# 维度
print(x.ndim)
# 模型
print(x.shape)

f = x + x1
print(np.maximum(f, 0.))
