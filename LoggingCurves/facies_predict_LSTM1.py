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
# data = (data - np.mean(data)) / np.std(data)  # 标准化数据

# 划分数据集
# train_data, test_data = train_test_split(data, test_size=0.2)


# 将数据划分为输入序列和目标序列
sequence_length = 100
input_sequence = []
target_sequence = []
for i in range(len(data) - sequence_length - 100):  # 减去100是为了预留100个点来作为预测目标
    input_sequence.append(data[i:i+sequence_length])
    target_sequence.append(data[i+sequence_length:i+sequence_length+100])

# 转换为张量
input_sequence = torch.tensor(input_sequence).unsqueeze(2).float()
target_sequence = torch.tensor(target_sequence).float()

# 将模型和输入数据移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_sequence = input_sequence.to(device)
target_sequence = target_sequence.to(device)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# 设置模型参数
input_size = 1
hidden_size = 64
output_size = 1

# 实例化模型并移动到GPU上
model = LSTMModel(input_size, hidden_size, output_size)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(input_sequence)
    loss = criterion(outputs, target_sequence)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 使用模型进行预测
model.eval()
with torch.no_grad():
    input_data = input_sequence[-1].unsqueeze(0).to(device)  # 取最后一个输入作为初始输入，并移动到GPU上
    predictions = []
    for _ in range(100):
        output = model(input_data)
        predictions.append(output.squeeze().tolist())  # 将张量转换为列表
        output = output.unsqueeze(1)
        # 将output用作下一步的输入
        input_data = torch.cat((input_data[:, 1:, :], output), dim=1)
        #out：1，1，100
        #intput：1，100，1

# 打印预测
print(predictions)

# 可视化预测结果
# plt.figure(figsize=(12,6))
# plt.plot(test_targets.squeeze().numpy(), label='Actual',linewidth=1)
# plt.plot(predicted, label='Predicted',linewidth=1)
# plt.xlabel('Time')
# plt.ylabel('Value')
#
# plt.legend()
# plt.show()