
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
import torch
import torch.nn as nn
import torch.optim as optim

sheet_name = 'Sheet2'

# 读取Excel文件中指定工作表的数据，从第三行开始读取数据，第一行作为特征名
df = pd.read_excel('../../data/dongtanALL.xlsx', sheet_name=sheet_name, header=0,skiprows=[1])
df = df.fillna(0)  # 将数据中的所有缺失值替换为0
# 删除最后一行数据 NaN
df = df.drop(df.index[-1])

# 假设 df 包含多个特征和名为 'target_feature' 的目标特征
features = df.drop(df.columns[[0, -1]], axis=1)
target = df[df.columns[0]]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)
# 创建一个 MinMaxScaler 对象
y_scaler = MinMaxScaler()
target = y_scaler.fit_transform(target.to_numpy().reshape(-1, 1))

# 构建滑动窗口样本
seq_length =5  # 滑动窗口大小，对应两小时的数据，假设每5分钟采集一次
pre_length = 1
X, y = [], []
for i in range(len(X_scaled) - seq_length - pre_length):
    seq = X_scaled[i:i+seq_length, :]
    label = target[i+seq_length:i+seq_length+pre_length]  # 预测未来一小时的数据，假设每5分钟一条
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

# 将数据转换成PyTorch张量
x_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()


# 划分数据集
train_size = int(0.7 * len(X_scaled))
val_size = int(0.1 * len(X_scaled))
test_size = len(X_scaled) - train_size - val_size

X_train, X_temp, y_train, y_temp = x_tensor[:train_size], x_tensor[train_size:train_size+val_size], y_tensor[:train_size], y_tensor[train_size:train_size+val_size]
X_val, X_test, y_val, y_test = x_tensor[train_size+val_size:], x_tensor[train_size+val_size:], y_tensor[train_size+val_size:], y_tensor[train_size+val_size:]

# 使用 DataLoader 封装数据
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


batch_size = 5
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)

        out = self.fc(out[:, -1, :])

        return out

# 实例化模型
input_size = 6  # 特征数量
hidden_size = 6
num_layers = 2
output_size = 1  # 预测未来一小时，每5分钟一条，共4条
model = GRU(input_size, hidden_size, num_layers, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练GRU模型
num_epochs = 1000
for epoch in range(num_epochs):
    for batch_x, batch_y in train_dataloader:
        # 前向传播
        outputs = model(batch_x)
        # print('````````````')
        # print(outputs)
        # print(batch_y)
        # 计算损失
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 验证GRU模型
model.eval()
with torch.no_grad():
    for val_batch_x, val_batch_y in val_dataloader:
        val_outputs = model(val_batch_x)
        val_loss = criterion(val_outputs, val_batch_y)
        print(f'Validation Loss: {val_loss.item():.4f}')

# 测试GRU模型
model.eval()
test_mse = 0.0
test_mae = 0.0
num_batches = 0
predictions = []  # 存储预测结果
true_values = []  # 存储真实值

with torch.no_grad():
    for test_batch_x, test_batch_y in test_dataloader:
        test_outputs = model(test_batch_x)
        print('true',test_batch_y.shape)
        print('pred',test_outputs.shape)
        test_loss = criterion(test_outputs, test_batch_y)
        test_mse += test_loss.item()

        # 计算MAE
        test_mae += torch.abs(test_outputs - test_batch_y).mean().item()

        # 存储预测结果（仅存储每个批次的第一个预测值）
        predictions.append(test_outputs[:, ].numpy())

        # 存储真实值（仅存储每个批次的第一个真实值）
        true_values.append(test_batch_y[:, ].numpy())

        num_batches += 1

# 计算平均MSE和MAE
avg_test_mse = test_mse / num_batches
avg_test_mae = test_mae / num_batches

print(f'Test MSE: {avg_test_mse:.4f}')
print(f'Test MAE: {avg_test_mae:.4f}')


# 将预测结果展平
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# def custom_function(x):
#     # return x*0.681 + 32.29 #point14
#     # return x*0.6 + 36.2 #point5
#     #return  x*0.8033+29.91#测点1
#     return  x*1.033+30.74#测点2
#     #return x * 0.5679 + 36.43  # 测点3
# # 绘制测试集真实值和预测值（仅绘制每个批次的第一个预测值）
# plt.figure(figsize=(8, 6))
# plt.plot(custom_function(true_values), label='True')
# plt.plot(custom_function(predictions), label='Predicted')
# plt.title('Comparison of True and Predicted Values (First Time Point)')
# plt.legend()
# plt.show()
