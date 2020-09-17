import tensorflow as tf

def sess_print(data):
    #
    sess = tf.InteractiveSession()
    print(data)
    print(data.eval())
    sess.close()

    # sess = tf.Session()
    # print(sess.run(data))
    # sess.close()


# 标量常量
t_1 = tf.constant(4)
# [1,3] 常量向量
t_2 = tf.constant([4, 3, 2])
# 要创建一个所有元素为零的张量，可以使用 tf.zeros() 函数。这个语句可以创建一个形如 [M，N] 的零元素矩阵，数据类型（dtype）可以是 int32、float32 等：
t_zero= tf.zeros([2, 3], tf.int32)

tf.ones = ''


if __name__ == '__main__':
    sess_print(t_zero)





