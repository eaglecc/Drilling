"""
__author__ = 'Cheng Yuchao'
__project__: GRU 大庆油田 进行测井曲线预测实验
__time__:  2023/10/27
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

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据
data = pd.read_csv('../../data/daqingyoutian/vertical_all_A1.csv')

data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['RMN-RMG', 'HAC     .us/m','GR      .   ',  'CAL     .cm ']].values
data_y = data['DEN     .g/cm3 '].values
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


# 2. 定义回看窗口大小
look_back = 500
future_window = 300
X, y = [], []
for i in range(len(data_x) - look_back - future_window + 1):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i + look_back: i + look_back + future_window])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
train_size = int(0.95 * len(X))
test_size = int(len(X) - train_size)
train_features = X[:train_size]
train_target = y[:train_size]
test_features = X[train_size:]
test_target = y[train_size:]
train_features = torch.FloatTensor(train_features).to(device)
train_target = torch.FloatTensor(train_target).to(device)
test_features = torch.FloatTensor(test_features).to(device)

# 定义批次大小
batch_size = 8  # 可以根据需求调整

# 使用DataLoader创建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_features, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 4. 定义GRU网络
# Replace the LSTMModel definition with a GRUModel
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, future_window)

    def forward(self, x):
        out, _ = self.gru(x)  # (batch_size, seq_length, hidden_size)
        out = self.dropout(out)
        out = out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, output_size)
        return out


# 5. 超参数
input_size = input_features  # 特征数量
hidden_size = 64
num_layers = 1
output_size = 1

# 6. Create the GRU model
model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
# num_epochs = 500
# for epoch in range(num_epochs):
#     # 前向传播
#     outputs = model(train_features)
#     loss = criterion(outputs, train_target)
#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 7.2 分batchsize进行训练
num_epochs = 20
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
predicted = predicted.cpu().numpy()
predicted1 = predicted[:, 0]


test_target_train = test_target[:, 0]
# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
plt.plot(test_target_train, label='True')
plt.plot(predicted1, label='Predicted')
plt.title('GRU测井预测')
plt.legend()
plt.show()

# 10. 计算深度
data_depth = data[['DEPT    .M ','DEN     .g/cm3 ']].values
data_depth = data_depth[look_back:]
data_depth = data_depth[:-future_window]
data_depth = data_depth[train_size - 1:]

# 11. 存储预测结果
# 反归一化
predicted_original_data = predicted1 * (max_value_y - min_value_y) + min_value_y
test_target_original_data = test_target_train * (max_value_y - min_value_y) + min_value_y
file_name = '../../result/wlp_transformer/daqingyoutian_GRU_DEN_result.xlsx'
df = pd.DataFrame({'depth': data_depth[:,0]})  # 创建一个新 DataFrame
df['gru_DEN_predicted'] = predicted_original_data
df['gru_DEN_true'] = test_target_original_data.flatten()
df.to_excel(file_name, index=False)  # index=False 防止写入索引列