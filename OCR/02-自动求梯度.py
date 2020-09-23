import tensorflow as tf

# # 自动求梯度
# x = tf.reshape(tf.Variable(range(4), dtype=tf.float32), (4, 1))
# print(x)
# with tf.GradientTape() as t:
#     t.watch(x)
#     y = 2 * tf.matmul(tf.transpose(x), x)
# dy_dx = t.gradient(y, x)
# print(dy_dx)
#
# # 训练模型和预测模型
#
# with tf.GradientTape(persistent=True) as g:
#     g.watch(x)
#     y = x * x
#     z = y * y
#     dz_dx = g.gradient(z, x)
#     dy_dx = g.gradient(y, x)
#
# print(dz_dx,dy_dx)
help(tf.norm)