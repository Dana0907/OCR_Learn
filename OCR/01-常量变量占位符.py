import tensorflow as tf


def sess_print(data):
    #
    sess = tf.InteractiveSession()
    print(data)
    print(data.eval())
    sess.close()
    return data
    # sess = tf.Session()
    # print(sess.run(data))
    # sess.close()


"""常量"""
# 标量常量
t_1 = tf.constant(4)
# [1,3] 常量向量
t_2 = tf.constant([4, 3, 2])
# 要创建一个所有元素为零的张量，可以使用 tf.zeros() 函数。这个语句可以创建一个形如 [M，N] 的零元素矩阵，数据类型（dtype）可以是 int32、float32 等：
t_zero = tf.zeros([2, 3], tf.int32)
# 要创建一个所有元素为一的张量
t_ones = tf.ones([2, 3], tf.int32)
# 在一定范围内生成一个从初值到终值等差排布的序列
t_linspace = tf.linspace(2.0, 1.0, 5)
t_range = tf.range(1, 10, 2)
# 具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组. 其中seed(随机种子，设置固定的随机数)
# 要在多次运行或会话中获得相同的随机数，应该将种子设置为一个常数值
# tf.set_random_seed(12)
t_random = tf.random_normal([3, 3], mean=2.0, stddev=4, seed=12)
# 要将给定的张量随机裁剪为指定的大小
t_random_1 = tf.random_crop(t_random, [2, 2], seed=12)
# 是想要重新排序的张量
t_shuffle = tf.random_shuffle(t_random)

"""变量"""

rand_t = tf.random_uniform([50, 50], 0, 10, seed=0)
# 变量通常在神经网络中表示权重和偏置。
# 设置默认值
t_a = tf.Variable(rand_t)
t_b = tf.Variable(rand_t)

weights = tf.Variable(tf.random_normal([100, 100], stddev=2))
bias = tf.Variable(tf.zeros([100]), name='biases')
weights2 = tf.Variable(weights.initial_value)
# 变量的值无法直接打印，须要先申明初始化
# 变量用print函数打印的只是变量的结构
se = tf.Session()
se.run(tf.global_variables_initializer())
print(se.run(weights))
print(se.run(weights2))

"""占位符"""

if __name__ == '__main__':
    pass
    # sess_print(rand_t)
