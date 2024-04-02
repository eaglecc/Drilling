"""
__author__ = 'Cheng Yuchao'
__project__: LSTM + CNN + 滑动窗口 进行岩性分类 ：方案2
__time__:  2024/3/29
__email__:"2477721334@qq.com"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 释放GPU缓存
torch.cuda.empty_cache()

# 导入数据
data = pd.read_csv('../data/train_data.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data.iloc[:, 4:9].values  # 特征列
data_y = data.iloc[:, 0].values  # 标签列
input_features = 5

data_y = F.one_hot(torch.from_numpy(data_y) - 1, num_classes=9).numpy()  # 假设有 9 种类别

min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值


# 归一化
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # 沿着列计算最小值
    max_vals = np.max(data, axis=0)  # 沿着列计算最大值
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


data_x = min_max_normalize(data_x)
# data_y = min_max_normalize(data_y)

# 2. 定义回看窗口大小
look_back = 500

X, y = [], []
for i in range(len(data_x) - look_back + 1):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i:i + look_back])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
X_train, X_test, y_train, test_target = train_test_split(X, y, test_size=0.2, random_state=42)
# 将数据转换为 PyTorch 张量
train_features = torch.FloatTensor(X_train).to(device)
train_target = torch.FloatTensor(y_train).to(device)
test_features = torch.FloatTensor(X_test).to(device)

# 定义批次大小
batch_size = 64  # 可以根据需求调整

# 使用DataLoader创建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_features, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 4. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # CNN 层
        self.conv1 = nn.Conv1d(hidden_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, output_size, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.3)
        # 全连接层
        self.fc = nn.Linear(32, output_size)  # 假设使用了池化层，所以特征数量变为原来的四分之一

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # (64 , 500 , 5) => (64 , 500 , 128)
        out = out.permute(0, 2, 1)  # (64 , 500 , 128) => (64 , 128, 500 )
        # CNN 层
        conv_out = self.conv1(out) # (64 , 128, 500 ) => (64 , 32, 500 )
        conv_out = nn.ReLU()(conv_out) # (64 , 32, 500 )
        conv_out = self.conv2(conv_out) # (64, 32, 500 ) => (64, 500, 9)
        conv_out = conv_out.permute(0, 2, 1)

        return conv_out


# 5. 超参数
input_size = input_features  # 特征数量
hidden_size = 128
num_layers = 2
num_classes = 9

# 6.创建LSTM模型、定义损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
# 7.1 不分batchsize进行训练
# num_epochs = 200
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(train_features)
#     loss = criterion(outputs, train_target)
#     # 反向传播和优化
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 7.2 分batchsize进行训练
num_epochs = 30
for epoch in range(num_epochs):
    for batch_features, batch_target in train_loader:
        # 将批次数据移到GPU上（如果可用）
        batch_features = batch_features.to(device)
        batch_target = batch_target.to(device)
        # 前向传播
        outputs = model(batch_features)
        loss = criterion(outputs, batch_target)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 8. 测试集预测
model.eval()
with torch.no_grad():
    predicted = model(test_features)
predicted = predicted[:, -1, :]
_, predicted = torch.max(predicted, 1)
predicted = predicted.cpu().numpy() + 1

labels = torch.argmax(torch.from_numpy(test_target[:, -1, :]), dim=1) + 1  # 获取独热编码中值为 1 的位置，并加一得到类别
labels = labels.cpu().numpy()

print(predicted)
print(labels)

# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
plt.plot(labels, label='True')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()


def accuracy_between_arrays(arr1, arr2):
    # 比较两个 ndarray 中元素是否相等，生成布尔值的 ndarray
    equal_elements = arr1 == arr2
    # 计算 True 的比例
    accuracy = np.mean(equal_elements)
    return accuracy


# 计算准确率
accuracy = accuracy_between_arrays(predicted, labels)
print("测试集准确率:", accuracy)

# 盲井上测试
data = pd.read_csv('../data/blind_data.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data.iloc[:, 4:9].values  # 特征列
data_y = data.iloc[:, -1].values  # 标签列


# 归一化
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # 沿着列计算最小值
    max_vals = np.max(data, axis=0)  # 沿着列计算最大值
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


data_x = min_max_normalize(data_x)

look_back = 500
X, y = [], []
for i in range(len(data_x) - look_back + 1):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i:i + look_back])
X = np.array(X)
y = np.array(y)
# 将数据转换为 PyTorch 张量
train_features = torch.FloatTensor(X).to(device)
train_target = torch.FloatTensor(y).to(device)

model.eval()
with torch.no_grad():
    predicted = model(train_features)
predicted = predicted[:, -1, :]
_, predicted = torch.max(predicted, 1)
predicted = predicted.cpu().numpy() + 1
labels = train_target[:, -1]
labels = labels.cpu().numpy()

print(predicted)
print(labels)
accuracy = accuracy_between_arrays(predicted, labels)
print("盲井准确率:", accuracy)
