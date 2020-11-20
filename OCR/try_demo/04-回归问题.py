from keras.datasets import boston_housing

# 创建波士顿房价数据集
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 查看波士顿房价数据
print(train_data.shape)
print(test_data.shape)
# print(train_targets)


# 数据标准化(归一化)
# 对于输入数据的每个特征（输入数据矩阵中的列），减去特征平均值，再除以标准差，这样得到的特征平均值为 0，标准差为 1。
# 平均值
mean = train_data.mean(axis=0)
# 标准差
std = train_data.std(axis=0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers


# 需要将同一个模型多次实例化
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(test_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer='rmsprop',
        loss='mse',  # 均方误差   预测值与目标值之差的平方。这是回归问题常用的损失函数
        metrics=['mae']  # 监控指标 预测值与目标值之差的绝对值
    )
    return model


# K折验证
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_score = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    # 准备训练数据
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    model = build_model()
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=1,
        verbose=0  # 静默模式
    )
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_score.append(val_mae)
    # print(history.history.keys())
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

print(all_score)
print(np.mean(all_score))
# print(all_mae_histories)
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
#
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
