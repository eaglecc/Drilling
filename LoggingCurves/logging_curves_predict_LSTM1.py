'''
使用LSTM预测测井GR曲线demo1
修改自facies_predict_LSTM2.py
设置look_back=3，即使用三个点预测下一个点
模型的outputsize设置为1，即每个时间步的预测值数量为1
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
import pickle
from sklearn.model_selection import train_test_split

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


pickle_file_path = '../data/facies_vectors.pickle'
training_data = read_pickle_file(pickle_file_path)
training_data['Well Name'] = training_data['Well Name'].astype('category')
training_data['Formation'] = training_data['Formation'].astype('category')
training_data['Well Name'].unique()

dataset = training_data.iloc[:, 4].values.astype(float)  # 提取第四列数据并转换 GR：伽马 为浮点数数组

# 数据预处理
dataset = (dataset - np.mean(dataset)) / np.std(dataset)  # 标准化数据

'''
这个函数是用于创建时间序列数据集的函数。它接受一个输入数据集和一个回溯值（look_back），
并返回两个数组，分别是输入数据（dataX）和对应的输出数据（dataY）。
'''
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

# 划分数据集
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size=0.2, random_state=42)

'''
RNN 读入的数据维度是 (seq, batch, feature)，所以要重新改变一下数据的维度
这里只有一个序列，所以 batch 是 1，
而输入的 feature 就是我们希望依据的几个月份，
这里我们定的是3个月份，所以 feature 就是3.
'''

train_X = train_X.reshape(-1, 1, 3)
train_Y = train_Y.reshape(-1, 1, 1)
#test_X = test_X.reshape(-1, 1, 3)

# reshape(-1, 1, 2)中的参数表示新的形状。其中：
# -1表示自动计算该维度的大小，以保持原始数据中的元素总数不变。
# 1表示在第二个维度上指定大小为1，即批次大小为1。
# 2表示在第三个维度上指定大小为2，即每个时间步骤的特征维度为2。

train_x = torch.from_numpy(train_X).to(device)
train_y = torch.from_numpy(train_Y).to(device)
#test_x = torch.from_numpy(test_X).to(device)


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.lstm(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


'''
定义模型并迁移到GPU:
输入的维度是 3，因为我们使用3个月的流量作为输入
隐藏层的维度可以任意指定，这里我们选的 64
'''
net = lstm_reg(3, 128).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(train_x).to(dtype=torch.float32).to(device)
    var_y = Variable(train_y).to(dtype=torch.float32).to(device)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    # 检查模型参数是否存在梯度
    if any(p.grad is not None for p in net.parameters()):
        optimizer.step()

    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data.item()))

net = net.eval()  # 转换成测试模式
data_X = data_X.reshape(-1, 1, 3)
data_X = torch.from_numpy(data_X).to(device)
var_data = Variable(data_X).to(dtype=torch.float32).to(device)
pred_test = net(var_data)  # 测试集的预测结果

# 将结果从GPU内存复制到主机内存并转换为NumPy数组
pred_test = pred_test.cpu().view(-1).detach().numpy()
# 画出实际结果和预测的结果
dataset = dataset[3:] #因为locckback设置的是3，故取第三个点之后的数据，对比展示预测结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
plt.show()
# 点个数太多了，减少点的个数，让图像更清晰
pred_dots = []
true_dots = []
for i in range(len(pred_test)):
    if(i % 50 == 0):
        pred_dots.append(pred_test[i])
        true_dots.append(dataset[i])
plt.plot(pred_dots, 'r', label='prediction')
plt.plot(true_dots, 'b', label='real')
plt.legend(loc='best')
plt.show()

# 计算均方误差（MSE）
mse = np.mean((pred_test - dataset) ** 2)
# 计算均方根误差（RMSE）
rmse = np.sqrt(np.mean((pred_test - dataset) ** 2))
# 计算平均绝对误差（MAE）
mae = np.mean(np.abs(pred_test - dataset))

print("LSTM Predict MSE:", mse)
print("LSTM Predict RMSE:", rmse)
print("LSTM Predict MAE:", mae)

# LSTM Predict MSE: 0.13752238050080318
# LSTM Predict RMSE: 0.37084010098801773
# LSTM Predict MAE: 0.22348642348413875