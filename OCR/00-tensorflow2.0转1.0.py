import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 紧急调用v1版本
# tf.compat.v1.disable_eager_execution()

# hello = tf.constant("hello tensorflow!")
# sess = tf.compat.v1.Session()
# print(sess.run(hello).decode())


v_1 = tf.constant([1, 2, 3, 4])
v_2 = tf.constant([2, 1, 5, 3])

v_add = tf.add(v_1, v_2)
# with tf.compat.v1.Session() as sess:
#     print(sess.run(v_add))
#     sess.close()

sess = tf.InteractiveSession()
print(v_add.eval())
sess.close()