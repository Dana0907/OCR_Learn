import tensorflow as tf

# print(tf.__version__)
x = tf.constant(range(12))
# print(x.shape)
# print(x)
# X = tf.TensorShape([12])
# print(X)

x = tf.reshape(x, (3, 4))
print(x)

y = tf.constant([[2, 1, 4, 5], [5, 2, 4, 1], [1, 4, 5, 2]])
print(y)

# z = tf.random.normal(shape=[3, 4], mean=0, stddev=1, seed=12)
# print(z)

# print(x+y)
# print(x*y)
# print(x/y)
# print(x//y)


# Y = tf.cast(y, tf.float32)
# print(tf.exp(Y))
#
#
#
# print(tf.matmul(x,tf.transpose(y)))
#
#
# print(tf.transpose(y))

print('*' * 50)
# 行链接  列连接
# print(tf.concat([x, y], axis=0), tf.concat([x, y], axis=1))
# print(tf.equal(x,y))
# print(tf.reduce_sum(x))
# print(tf.reduce_mean(x))

# X = tf.cast(x, tf.float32)
# print(tf.norm(X))

# 广播机制
# a = tf.reshape(tf.constant(range(3)), (3, 1))
# b = tf.reshape(tf.constant(range(2)), (1, 2))
#
# print(a, b)
# print(a+b)
# print(a*b)

# 索引
# 赋值
# x = tf.Variable(x)
# x[0, 0].assign(9)
# print(x)

# x = tf.Variable(x)
# print(x[1:2, :].assign(tf.zeros(x[1:2, :].shape, dtype=tf.int32)))

# x = tf.Variable(x)
# y = tf.cast(y,dtype=tf.int32)
#
# before = id(y)
# print(before)
# y = x + y
# print(id(y))


# z = tf.Variable(tf.zeros_like(y))
# #
# # print(id(z))
# # print(z)
# # z[:].assign(x+y)
# #
# # print(id(z))


import numpy as np

P = np.ones((2, 3))
print(P)
D = tf.constant(P)
print(D)
print(np.array(D))