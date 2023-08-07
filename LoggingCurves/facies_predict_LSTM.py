import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
使用LSTM进行测井曲线预测
其中训练集、测试集为随机分类，每次运行划分不一样，因此生成的测井曲线也不一样
最后可视化展示测试集的训练效果
'''

# 读取pickle包
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


pickle_file_path = '../data/facies_vectors.pickle'
training_data = read_pickle_file(pickle_file_path)
# The 'Well Name' and 'Formation' columns can be turned into a categorical data type.
# 这三行代码涉及对 Pandas 数据框中的某些列进行数据类型转换和唯一值的提取。
training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()

# # 准备数据
# input_sequence = training_data.iloc[:, 4].values  # 第一列作为输入序列
# target_sequence = training_data.iloc[:, 4].values  # 第一列作为输入序列
#
# # 转换为Tensor
# input_sequence = torch.tensor(input_sequence[:-100], dtype=torch.float32).view(-1, 1, 1)  #使用view(-1, 1, 1) 对新创建的张量进行形状变换，将其转换为三维张量。
# target_sequence = torch.tensor(target_sequence[-100:], dtype=torch.float32).view(-1, 1, 1)

# 提取并转换 GR：伽马 为浮点数数组
data = training_data.iloc[:, 4].values.astype(float)
data = (data - np.mean(data)) / np.std(data)  # 标准化数据

# 划分数据集
train_data, test_data = train_test_split(data, test_size=0.2)

# 数据预处理:data是一个一维数组，表示原始的时间序列数据。window_size是指定的窗口大小，用于定义每个训练样本的输入窗口长度。
def prepare_data(data, window_size):
    windows = []
    targets = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    windows = torch.tensor(windows).unsqueeze(2).float()  # 添加维度以匹配LSTM模型的输入要求
    targets = torch.tensor(targets).unsqueeze(1).float()
    return windows, targets

window_size = 50
train_windows, train_targets = prepare_data(train_data, window_size)
test_windows, test_targets = prepare_data(test_data, window_size)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 创建LSTM模型实例
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 300
batch_size = 32

for epoch in range(num_epochs):
    permutation = torch.randperm(train_windows.size(0))
    for i in range(0, train_windows.size(0), batch_size): #循环遍历训练集，以批次大小为步长进行迭代。
        indices = permutation[i:i + batch_size] #根据随机排列的索引，选择当前批次的样本索引。
        batch_windows = train_windows[indices].to(device)
        batch_targets = train_targets[indices].to(device)

        model.zero_grad()
        outputs = model(batch_windows)
        loss = criterion(outputs, batch_targets)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 使用训练好的模型进行预测
model.eval() #将模型切换到评估模式
with torch.no_grad():
    test_windows = test_windows.to('cuda')
    predicted = model(test_windows).cpu().squeeze().numpy()

# 可视化预测结果
plt.figure(figsize=(12,6))
plt.plot(test_targets.squeeze().numpy(), label='Actual',linewidth=1)
plt.plot(predicted, label='Predicted',linewidth=1)
plt.xlabel('Time')
plt.ylabel('Value')

plt.legend()
plt.show()