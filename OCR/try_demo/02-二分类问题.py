from keras.datasets import imdb
import numpy as np

# 定义训练数据
# num_words = 10000 仅保留训练数据中前10000个最常出现的单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 定义层
# 选择损失函数 优化器 监控指标
# 调用模型中的fit方法在训练数据上进行迭代

# print(test_data[1])


max_train = [max(sequence) for sequence in train_data]
print(max(max_train))
print(len(max_train))

word_index = imdb.get_word_index()
# 字典key&value互换
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])

# 将某条评论迅速解码成英文单纯
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# print(decoded_review)

# 将整数学列编码成二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
    result = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        # print(sequence)
        result[i, sequence] = 1

    return result


# 将测试数据向量化
# print(train_data)
x_train = vectorize_sequences(train_data)
# print(x_train)
x_test = vectorize_sequences(test_data)

# 标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')



# 留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 模型定义
from keras import models
from keras import layers

# model = models.Sequential()
#
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # 编译模型
# model.compile(
#     optimizer='rmsprop',
#     loss='binary_crossentropy',
#     metrics=['acc']
#
# )



# history = model.fit(
#     partial_x_train,
#     partial_y_train,
#     epochs=20,
#     batch_size=512,
#     validation_data=(x_val, y_val)
#
# )
# # history_dict = history.history
# # print(history_dict.keys())
# # 输出结果
# result = model.evaluate(x_test, y_test)
# print(result)
# # 回执训练损失和验证损失
# import matplotlib.pyplot as plt
#
# history_dict = history.history
#
# # 绘制训练损失和验证损失
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values) + 1)
# # 蓝色圆点
# plt.plot(epochs, loss_values, 'bo', label='Training loss')
# # 蓝色实线
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
#
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 绘制训练精度和验证精度
# plt.clf()  # 清空图像
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

md2 = models.Sequential()
md2.add(layers.Dense(16, activation='tanh', input_shape=(10000,)))
md2.add(layers.Dense(16, activation='tanh'))
md2.add(layers.Dense(1, activation='sigmoid'))

md2.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['acc'])

md2.fit(
    x_train, y_train,
    epochs=4,
    batch_size=512
)

result2 = md2.evaluate(x_test, y_test)
print(result2)

predict = md2.predict(x_train)
print(predict)