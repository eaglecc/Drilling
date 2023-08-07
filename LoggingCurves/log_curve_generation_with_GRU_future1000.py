import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm

'''
    GRU预测伽马测井曲线未来1000个数据点
    
'''


class Config():
    data_path = '../data/facies_vectors.csv'
    timestep = 1  # 时间步长，就是利用多少时间窗口
    batch_size = 32  # 批次大小
    feature_size = 1  # 每个步长对应的特征数量，这里只使用1维，每天的风速
    hidden_size = 256  # 隐层大小
    output_size = 1  # 由于是单输出任务，最终输出层大小为1，预测未来1天风速
    num_layers = 2  # gru的层数
    epochs = 10  # 迭代轮数
    best_loss = 0  # 记录损失
    learning_rate = 0.0003  # 学习率
    model_name = 'gru'  # 模型名称
    save_path = './{}.pth'.format(model_name)  # 最优模型保存路径


config = Config()

# 1.加载时间序列数据
df = pd.read_csv(config.data_path)
df = df.iloc[:, 4:9]
# 2.将数据进行标准化
scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['GR']).reshape(-1, 1))


# 形成训练数据，例如12345789 12-3456789
def split_data(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 将整个窗口的数据保存到X中，将未来一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep][:, 0])
        dataY.append(data[index + timestep][0])
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]


# 3.获取训练数据
x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.feature_size)

# 4.将数据转为tensor
x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

# 5.形成训练数据集
train_data = TensorDataset(x_train_tensor, y_train_tensor)
test_data = TensorDataset(x_test_tensor, y_test_tensor)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data,
                                           config.batch_size,
                                           False)

test_loader = torch.utils.data.DataLoader(test_data,
                                          config.batch_size,
                                          False)


# 7.定义GRU网络
class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden

        # GRU运算
        output, h_0 = self.gru(x, h_0)

        # 获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape

        # 将output变成 batch_size * timestep, hidden_dim
        output = output.reshape(-1, hidden_size)

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 转换维度，用于输出
        output = output.reshape(timestep, batch_size, -1)

        # 我们只需要返回最后一个时间片的数据即可
        return output[-1]


model = GRU(config.feature_size, config.hidden_size, config.num_layers, config.output_size)  # 定义GRU网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)  # 定义优化器

# 8.模型训练
for epoch in range(config.epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 config.epochs,
                                                                 loss)

    # 模型验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(model.state_dict(), config.save_path)

print('Finished Training')

# 9.绘制结果
'''
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[: plot_size]).reshape(-1, 1)), "b"):
1. model(x_train_tensor)：使用训练好的模型 model 对输入数据 x_train_tensor 进行预测，得到预测结果。
2. detach().numpy()：将预测结果从 PyTorch 的张量转换为 NumPy 数组，以便后续处理。
3. [: plot_size]：对预测结果进行切片操作，仅选择前 plot_size 个数据点进行绘制。
4. reshape(-1, 1)：将预测结果的形状调整为一列数据的形式，以适应绘图函数的要求。
5. scaler.inverse_transform(...)：使用之前定义的缩放器模型 scaler 将缩放后的预测结果逆转为原始数据的尺度。
6. "b"：指定折线图的颜色为蓝色。

plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
1. y_train_tensor：训练数据的目标值，通常表示为标签或真实值。
2. detach().numpy()：将目标值从 PyTorch 的张量转换为 NumPy 数组。
3. reshape(-1, 1)：将目标值的形状调整为一列数据的形式，以适应绘图函数的要求。
4. [: plot_size]：对目标值进行切片操作，仅选择前 plot_size 个数据点进行绘制。
5. scaler.inverse_transform(...)：使用之前定义的缩放器模型 scaler 将缩放后的目标值逆转为原始数据的尺度。
6. "r"：指定折线图的颜色为红色。
'''

fulture_size = 100
for _ in range(fulture_size):
    predict_train_data = model(x_train_tensor)
    last_data = predict_train_data[-1].unsqueeze(1)
    # 将最后一条数据添加到 last_data 的末尾
    x_train_tensor = torch.cat((x_train_tensor, last_data.unsqueeze(0)), dim=0)

plot_size = 500
plt.figure(figsize=(12, 8))
plt.rc('font', family='FangSong', weight='bold')  # 设置中文可见
plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy()[-(plot_size+fulture_size):]).reshape(-1, 1)), "b")
plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[-plot_size:]), "r")
plt.title("训练集预测结果")
plt.show()

# y_test_pred = model(x_test_tensor)
# plt.figure(figsize=(12, 8))
# plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[: plot_size]), "b")
# plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
# plt.title("测试集预测结果")
# plt.show()
