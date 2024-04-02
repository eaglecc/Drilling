# https://www.bilibili.com/video/BV1Ez4y1E7VJ/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=4517f99a287ad84d391467c4eb7a0737
"""
__author__ = 'Cheng Yuchao'
__project__: LSTM 进行岩性分类 ：方案2
__time__:  2024/3/29
__email__:"2477721334@qq.com"
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# 假设你已经有了训练数据 X_train, y_train 以及测试数据 X_test, y_test
data = pd.read_csv('../data/train_data.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0

# 假设数据集的最后一列是标签列，其他列是特征列
X = data.iloc[:, 4:9].values  # 特征列
y = data.iloc[:, 0].values  # 标签列
# 将标签数据转换为 PyTorch 的张量
y_tensor = torch.from_numpy(y)
# 将类别索引转换为独热编码
y = F.one_hot(y_tensor - 1, num_classes=9).numpy()  # 假设有 9 种类别

# 归一化
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # 沿着列计算最小值
    max_vals = np.max(data, axis=0)  # 沿着列计算最大值
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data
X = min_max_normalize(X)


# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 训练模型
def train_model(model, X_train, y_train, num_epochs, batch_size):
    loss_list = []
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = torch.from_numpy(X_train[i:i + batch_size].astype(np.float64)).float().unsqueeze(1)
            labels = torch.from_numpy(y_train[i:i + batch_size]).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}],Loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    return loss_list


# 评估
def evaluate_model(model, X, y, name):
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X.astype(np.float64)).float().unsqueeze(1)
        # labels = torch.from_numpy(y).long()
        labels = torch.argmax(torch.from_numpy(y), dim=1) + 1  # 获取独热编码中值为 1 的位置，并加一得到类别
        outputs = model(inputs)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted + 1 == labels).sum().item() / labels.size(0)

    print(' {} Accuracy: {:.2f}%'.format(name, accuracy * 100))
    print("labels:", labels)
    print("predicted:", predicted + 1)
    return labels, predicted + 1


# 绘制损失曲线
def plot_loss(loss_list):
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.show()


# 定义模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        output = self.fc(h_n[-1])
        return output.squeeze()  # 去除维度为1的维度


# 设置参数
input_size = X_train.shape[1]
output_size = 9
hidden_size = 128
num_epochs = 300
batch_size = 32

# 实例化模型
model = LSTMClassifier(input_size, hidden_size, output_size)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练
loss_list = train_model(model, X_train, y_train, num_epochs, batch_size)
# 评估测试集
labels, predicted = evaluate_model(model, X_test, y_test, "X_test")
# 绘图
plot_loss(loss_list)
# 绘制混淆矩阵
