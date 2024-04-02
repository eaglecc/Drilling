"""
__author__ = 'Cheng Yuchao'
__project__: LSTM + CNN 进行岩性分类 ：方案1
__time__:  2024/3/29
__email__:"2477721334@qq.com"
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# 1. 准备数据集
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


# 转换为PyTorch的Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. 定义数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. 定义LSTM + CNN 模型
class LSTMCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMCNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=32, kernel_size=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        #self.fc = nn.Linear(32 * ((input_size-1)//2), num_classes)  # input_size-2 是为了保持特征维度一致
        self.fc = nn.Linear(32, num_classes)  # input_size-2 是为了保持特征维度一致

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(0, 2, 1)  # 将输出的维度调整为 (batch_size, hidden_size, seq_len)
        # CNN layer
        out = self.conv1d(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        # Flatten
        out = self.fc(out)
        return out

input_size = 5
hidden_size = 128
num_layers = 2
num_classes = 9

model = LSTMCNNClassifier(input_size, hidden_size, num_layers, num_classes)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 评估模型
model.eval()
with torch.no_grad():
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        labels = torch.argmax(labels, dim=1) + 1  # 获取独热编码中值为 1 的位置，并加一得到类别
        outputs = model(inputs.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)

        correct_train += (predicted + 1 == labels).sum().item()

    print(f'Training Accuracy: {(100 * correct_train / total_train):.2f}%')

    correct_test = 0
    total_test = 0
    for inputs, labels in test_loader:
        labels = torch.argmax(labels, dim=1) + 1  # 获取独热编码中值为 1 的位置，并加一得到类别
        outputs = model(inputs.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted + 1 == labels).sum().item()

    print(f'Testing Accuracy: {(100 * correct_test / total_test):.2f}%')

# 盲井上测试
data = pd.read_csv('../data/blind_data.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0

# 假设数据集的最后一列是标签列，其他列是特征列
X = data.iloc[:, 4:9].values  # 特征列
y = data.iloc[:, -1].values  # 标签列
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

# 转换为PyTorch的Tensor
X_train_tensor = torch.tensor(X, dtype=torch.float32)
y_train_tensor = torch.tensor(y, dtype=torch.long)

# 定义数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
model.eval()
with torch.no_grad():
    correct_train = 0
    total_train = 0
    for inputs, labels in train_loader:
        labels = torch.argmax(labels, dim=1) + 1  # 获取独热编码中值为 1 的位置，并加一得到类别
        outputs = model(inputs.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted + 1 == labels).sum().item()

    print(f'盲井 Training Accuracy: {(100 * correct_train / total_train):.2f}%')


# Stack ；1
# Training Accuracy: 54.72%
# Testing Accuracy: 50.60%
# 盲井 Training Accuracy: 43.61%

# Stack ；2
# Training Accuracy: 59.63%
# Testing Accuracy: 55.30%
# 盲井 Training Accuracy: 36.51%