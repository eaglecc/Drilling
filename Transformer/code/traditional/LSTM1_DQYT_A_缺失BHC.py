"""
__author__ = 'Cheng Yuchao'
__project__: LSTM进行大庆油田 A井上  缺失测井曲线预测实验  DEN
__time__:  2024/2/29
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


warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 释放GPU缓存
torch.cuda.empty_cache()

# 导入数据
data = pd.read_csv('../../data/daqingyoutian/vertical_all_A1.csv')
# data.dropna(axis=0, how='any')  #只要行中包含任何一个缺失值，就删除整行。
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
#data_x = data[['RMN-RMG', 'HAC     .us/m','GR      .   ', 'SP      .mv  ',  'CAL     .cm ', 'DEN     .g/cm3 ']].values
data_x = data[['RMN-RMG', 'HAC     .us/m' ,'SP      .mv  ',  'CAL     .cm ']].values
data_y = data['BHC     .'].values
input_features = 4

min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
# 归一化
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # 沿着列计算最小值
    max_vals = np.max(data, axis=0)  # 沿着列计算最大值
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data
data_x = min_max_normalize(data_x)
data_y = min_max_normalize(data_y)

# data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
# data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 2. 定义回看窗口大小
look_back = 500

X, y = [], []
for i in range(len(data_x) - look_back + 1):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i:i + look_back])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
train_size1 = int(0.4 * len(X))
train_size2 = int(0.6 * len(X))

train_features1 = X[:train_size1]
train_features2 = X[train_size2:]
train_features = np.concatenate((train_features1, train_features2), axis=0)

train_target1 = y[:train_size1]
train_target2 = y[train_size2:]
train_target = np.concatenate((train_target1, train_target2), axis=0)
test_features = X[train_size1:train_size2]
test_target = y[train_size1:train_size2]

train_features = torch.FloatTensor(train_features).to(device)
train_target = torch.FloatTensor(train_target).to(device)
test_features = torch.FloatTensor(test_features).to(device)

# 定义批次大小
batch_size = 8  # 可以根据需求调整

# 使用DataLoader创建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_features, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 4. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # （3050，50，6）-->（3050，50，64）
        out = self.dropout(out)
        # out = out[:, -1, :]  # 取最后一个时间步的输出（3050，50，64) --> (3050，64)
        out = self.fc(out)  # （3050，64） --> (3050 , 1)
        return out


# 5. 超参数
input_size = 4  # 特征数量
hidden_size = 64
num_layers = 1
output_size = 1

# 6.创建LSTM模型、定义损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
# 7.1 不分batchsize进行训练
# num_epochs = 100
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(train_features)
#     outputs = outputs.squeeze(dim=-1)
#     loss = criterion(outputs, train_target)
#     # 反向传播和优化
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 7.2 分batchsize进行训练
num_epochs = 80
for epoch in range(num_epochs):
    for batch_features, batch_target in train_loader:
        # 将批次数据移到GPU上（如果可用）
        batch_features = batch_features.to(device)
        batch_target = batch_target.to(device)
        # 前向传播
        outputs = model(batch_features)
        outputs = outputs.squeeze(dim=-1)
        loss = criterion(outputs, batch_target)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 8. 测试集预测
model.eval()
with torch.no_grad():
    predicted = model(test_features)
predicted = predicted.squeeze(dim=-1)
predicted = predicted.cpu().numpy()
predicted = predicted[:, 0]

# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
plt.plot(test_target[:, 0], label='True')
plt.plot(predicted, label='Predicted')
plt.title('LSTM测井预测')
plt.legend()
# 使用savefig保存图表为文件
# plt.savefig(('../../result/lstm/epoch_{}.png').format(num_epochs))  # 保存为PNG格式的文件
# print("图片已经存储至：../../result/lstm/")
plt.show()


predicted_original_data = predicted * (max_value_y - min_value_y) + min_value_y
test_target_original_data = test_target[:, 0] * (max_value_y - min_value_y) + min_value_y
file_name = '../../result/wlp_transformer/缺失预测A_lstm_BHC_result.xlsx'
df = pd.DataFrame()  # 创建一个新 DataFrame
df['lstm_BHC_predicted'] = predicted_original_data
df['lstm_BHC_true'] = test_target_original_data.flatten()
df.to_excel(file_name, index=False)  # index=False 防止写入索引列