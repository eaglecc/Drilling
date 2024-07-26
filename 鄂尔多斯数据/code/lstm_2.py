# @Time    : 2024/5/26 9:24
# @Author  : ChengYuchao
# @Email   : 2477721334@qq.com
# @Project :"大66井", "大62井" 作为盲井进行测试、其余井作为训练井

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 准备数据集
file_path = '../data/物性分析平均值的计算.xlsx'
sheet1_name = '盒2'
sheet2_name = '不同沉积相物性对比'
category_column = '井号'
columns_to_select = ['井号', '井深', '层位', '孔隙度', '渗透率', '碳酸盐', '密度']

def extract_category_data(file_path, sheet1_name, sheet2_name, category_column, columns_to_select):
    # 读取Excel文件中的数据
    # df1 = pd.read_excel(file_path, sheet_name=sheet1_name)
    df2 = pd.read_excel(file_path, sheet_name=sheet2_name)
    category_counts = df2[category_column].value_counts()
    category_data_dict = {}
    # 遍历category_counts
    for category, count in category_counts.items():
        category_data = df2[df2[category_column] == category]
        category_data_dict[category] = category_data[columns_to_select]

    return category_data_dict

extracted_data = extract_category_data(file_path, sheet1_name, sheet2_name, category_column, columns_to_select)

data_66 = extracted_data['大49井']
data_62 = extracted_data['大39井']
test_data = pd.concat([data_66, data_62], ignore_index=True)


exclude_keys = {"大49井", "大39井"}
filtered_dfs = [df for key, df in extracted_data.items() if key not in exclude_keys]
result_df = pd.concat(filtered_dfs, ignore_index=True)

data = result_df.fillna(0)  # 将数据中的所有缺失值替换为0

# 3. 异常检测(Z-Score)、平滑处理
z_score_data = data.loc[:, ['井深','孔隙度', '渗透率', '密度']]
z_scores = np.abs((z_score_data - z_score_data.mean()) / z_score_data.std())
# 设置一个阈值，通常为3
threshold = 3
outliers = z_scores > threshold
rows_to_remove = outliers.any(axis=1)
# 删除含有异常值的行
data = data[~rows_to_remove].reset_index(drop=True)
# x = data['井深']
plt.figure(figsize=(12, 8))
plt.plot(data['孔隙度'], label='孔隙度')
plt.plot(data['渗透率'], label='渗透率')
plt.plot(data['密度'], label='密度')
plt.xlabel('序号')
plt.ylabel('数值')
plt.title('井深与孔隙度、渗透率、密度的关系')
plt.legend()
plt.grid(True)
plt.show()

# 4. 使用LSTM进行预测
data_x = data[['渗透率', '孔隙度', '密度']].values
data_y = data['渗透率'].values
input_features = 3

# 归一化
min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 定义回看窗口大小
look_back = 10
future_window = 1
X, y = [], []
for i in range(len(data_x) - look_back - future_window + 1):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i + look_back: i + look_back + future_window])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
# 顺序划分
train_size = int(1 * len(X))
train_features = X[:train_size]
train_target = y[:train_size]
train_features = torch.FloatTensor(train_features).to(device)
train_target = torch.FloatTensor(train_target).to(device)

test_data = test_data.fillna(0)  # 将数据中的所有缺失值替换为0
test_data_x = test_data[['渗透率', '孔隙度', '密度']].values
test_data_y = test_data['渗透率'].values
min_value_y = test_data_y.min()  # 训练时y的最小值
max_value_y = test_data_y.max()  # 训练时y的最大值
test_data_x = (test_data_x - test_data_x.min()) / (test_data_x.max() - test_data_x.min())
test_data_y = (test_data_y - min_value_y) / (max_value_y - min_value_y)
test_X, test_y = [], []
for i in range(len(test_data_x) - look_back - future_window + 1):
    test_X.append(test_data_x[i:i + look_back])
    test_y.append(test_data_y[i + look_back: i + look_back + future_window])
test_features = np.array(test_X)
test_features = torch.FloatTensor(test_features).to(device)
test_target = np.array(test_y)


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
        self.fc = nn.Linear(hidden_size, future_window)

    def forward(self, x):
        out, _ = self.lstm(x)  # （3050，50，6）-->（3050，50，64）
        out = self.dropout(out)
        out = out[:, -1, :]  # 取最后一个时间步的输出（3050，50，64) --> (3050，64)
        out = self.fc(out)  # （3050，64） --> (3050 , 200)
        return out


# 5. 超参数
input_size = input_features  # 特征数量
hidden_size = 64
num_layers = 1
output_size = 1

# 6.创建LSTM模型、定义损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 训练模型
num_epochs = 200
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
predicted = predicted[:, 0]
test_target = test_target[:, 0]
# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
plt.plot(test_target, label='True')
plt.plot(predicted, label='Predicted')
plt.title('LSTM测井预测')
plt.legend()
plt.show()

# 10. Calculate RMSE、MAPE
mse = np.mean((test_target - predicted) ** 2)
rmse = np.sqrt(np.mean((test_target - predicted) ** 2))
mape = np.mean(np.abs((test_target - predicted) / test_target))
mae = np.mean(np.abs(test_target - predicted))
print("MSE", mse)
print("MAE", mae)
print("RMSE", rmse)
print("MAPE:", mape)
