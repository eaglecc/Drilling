"""
__author__ = 'Cheng Yuchao'
__project__: LSTM进行测井曲线预测实验
__time__:  2023/09/28
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

# 导入数据
data = pd.read_csv('../../data/Well3_EPOR0_1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
# DataSet2 = data[['GR', 'DENSITY', 'VSHALE', 'PEF', 'DPHI', 'EPOR0', 'LITH', 'DEPTH' , 'NPHI']]
# DataSet1 = data[['GR', 'NPHI', 'DENSITY', 'PEF', 'EPOR0', 'DEPTH']]

data_x = data[['GR', 'NPHI', 'PEF', 'EPOR0']]
data_y = data['DENSITY']
# data_y = data['DENSITY']
# data_y = data['EPOR0']

# 归一化
min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 2. 定义回看窗口大小
look_back = 80
# 创建回看窗口数据
X, y = [], []
for i in range(len(data_x) - look_back):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i + look_back])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
train_size1 = int(0.5 * len(X))
train_size2 = int(0.7 * len(X))

train_features1 = X[:train_size1]
train_features2 = X[train_size2:]
train_features = np.concatenate((train_features1, train_features2), axis=0)

train_target1 = y[:train_size1]
train_target2 = y[train_size2:]
train_target = np.concatenate((train_target1, train_target2), axis=0)
test_features = X[train_size1:train_size2]
test_target = y[train_size1:train_size2]

train_features = torch.FloatTensor(train_features).to(device)
train_target = torch.FloatTensor(train_target).view(-1, 1).to(device)
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
        out = out[:, -1, :]  # 取最后一个时间步的输出（3050，50，64) --> (3050，64)
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
# num_epochs = 400
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
num_epochs = 10
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
    predicted = model(torch.FloatTensor(X).to(device))
predicted = predicted.cpu().numpy()

# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
plt.plot(y, label='True')
plt.plot(predicted, label='Predicted')
plt.title('LSTM测井预测')
plt.legend()
# 使用savefig保存图表为文件
# plt.savefig(('../../result/lstm/epoch_{}.png').format(num_epochs))  # 保存为PNG格式的文件
# print("图片已经存储至：../../result/lstm/")
plt.show()

# 10. Calculate RMSE、MAPE
mse = np.mean((y - predicted) ** 2)
rmse = np.sqrt(np.mean((y - predicted) ** 2))
mape = np.mean(np.abs((y - predicted) / y))
mae = np.mean(np.abs(y - predicted))
print("MSE", mse)
print("MAE", mae)
print("RMSE", rmse)
print("MAPE:", mape)

# 11. 存储预测结果
# file_name = '../../result/lstm_result.xlsx'
# # 检查文件是否存在
# if os.path.exists(file_name):
#     # 如果文件已存在，读取已有数据
#     df = pd.read_excel(file_name)
#     # 创建一个新列并将数据添加到 DataFrame
#     df['lstm_DENSITY_predicted'] = predicted
#     df['lstm_DENSITY_true'] = y
#     # 写入 DataFrame 到 Excel 文件
#     df.to_excel(file_name, index=False)  # index=False 防止写入索引列
# else:
#     # 如果文件不存在，创建一个新 Excel 文件并存储数据
#     df = pd.DataFrame({'lstm_DENSITY_predicted': predicted.flatten()})  # 创建一个新 DataFrame
#     df['lstm_DENSITY_true'] = y
#     df.to_excel(file_name, index=False)  # index=False 防止写入索引列
