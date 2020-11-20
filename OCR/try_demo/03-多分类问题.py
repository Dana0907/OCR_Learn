from keras.datasets import reuters

# 定义训练数据
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))

print(train_data[10])

# 将索引解码为新闻文本
# word_index = reuters.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_newswire = ' '.join(reverse_word_index.get(i-3,'?') for i in train_data[1])
# print(decoded_newswire)
# 编码数据
import numpy as np


# 向量序列化（向量转换问浮点数张量）
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.
    return results


# 将训练测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 将训练测试的标签向量化

# 方法一
# one_hot_train_labels = vectorize_sequences(train_labels, dimension=46)
# one_hot_test_labels = vectorize_sequences(test_labels, dimension=46)

# 方法二   Keras内置方法
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

# 定义模型
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))




# 编译模型
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',   # 浮点数标签使用分类交叉熵，对于整数标签，你应该使用sparse_categorical_crossentropy
    metrics=['accuracy']
)

# 留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# 绘制训练损失和验证损失
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 清空图像
plt.clf()

# 绘制训练精度和验证精度
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#  完全随机分类 精度
# import copy
# test_labels_copy = copy.copy(test_labels)
# np.random.shuffle(test_labels_copy)
# hits_array = np.array(test_labels) == np.array(test_labels_copy)
# result = float(np.sum(hits_array)) / len(test_labels)
# print(result)





# 在新数据上生成预测结果
# 预言
# prediction = model.predict(x_test)
# print(prediction[0].shape)
# #
# print(np.sum(prediction[0]))
# print(np.argmax(prediction[0]))


