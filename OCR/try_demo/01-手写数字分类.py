from keras.datasets import mnist

# 导入数据 mnist.npz  放置 在 C:\Users\Administrator\.keras\datasets\
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 训练数据
print(train_images.shape)
print(test_images.ndim)
print(train_labels)
print(len(train_labels))

# 测试数据
print(test_images.shape)

print(test_labels)
print(len(test_labels))



# digit = train_images[8]
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

from keras import models
from keras import layers




# 全连接
network = models.Sequential()
# 第一层 activation激活函数
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# 第二层
network.add(layers.Dense(10, activation='softmax'))

# 编译 optimizer 优化器  loss 损失函数  metrics 监控指标
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# 准备图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 准备标签 (分类)
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)




if __name__ == '__main__':
    pass

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print('loss:', test_loss)
    print('acc:', test_acc)
