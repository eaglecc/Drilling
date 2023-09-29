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


warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据
data = pd.read_csv('../data/Well1_EPOR0_1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['NPHI', 'DENSITY', 'VSHALE', 'DPHI', 'EPOR0', 'LITH']]
data_y = data['GR']

# 归一化
min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 2. 定义回看窗口大小
look_back = 50
# 创建回看窗口数据
X, y = [], []
for i in range(len(data_x) - look_back):
    X.append(data_x[i:i+look_back])
    y.append(data_y[i+look_back])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
train_size = int(0.8 * len(X))
train_features = X[:train_size]
train_target = y[:train_size]
test_features = X[train_size:]
test_target = y[train_size:]

train_features = torch.FloatTensor(train_features).to(device)
train_target = torch.FloatTensor(train_target).view(-1, 1).to(device)
test_features = torch.FloatTensor(test_features).to(device)

# 4. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x) # （3050，50，6）-->（3050，50，64）
        out = out[:, -1, :] # （3050，50，64) --> (3050，64)
        out = self.fc(out)  # 取最后一个时间步的输出（3050，64） --> (3050 , 1)
        return out

# 5. 超参数
input_size = 6  # 特征数量
hidden_size = 64
num_layers = 1
output_size = 1

# 6.创建LSTM模型、定义损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(train_features)
    loss = criterion(outputs, train_target)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 8. 测试集预测
with torch.no_grad():
    predicted = model(test_features)
predicted = predicted.cpu().numpy()

# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
plt.plot(test_target, label='True')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()