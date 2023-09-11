"""
__author__ = 'Cheng Yuchao'
__project__:滑动窗口步幅为1
__time__:  2023/09/11
__email__:"2477721334@qq.com"
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data as Data
import math
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据
data = pd.read_csv('../data/Well1_EPOR0_1.csv')
# data.dropna(axis=0, how='any')  #只要行中包含任何一个缺失值，就删除整行。
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['DENSITY', 'NPHI', 'VSHALE', 'DPHI', 'EPOR0']].values
data_y = data['GR'].values

# Z-Score归一化 z = (x - mean) / std
# data_x_normalized = (data_x - data_x.mean()) / data_x.std()
# data_y_normalized = (data_y - data_y.mean()) / data_y.std()

#  Min-Max归一化
# min_value_y = data_y.min()  # 训练时y的最小值
# max_value_y = data_y.max()  # 训练时y的最大值
# data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
# data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 四个数据划分为一组，用前三个数据预测后一个
data_4_x = []
data_4_y = []

for i in range(0, len(data_y) - 4, 1):
    data_4_x.append(data_x[i:i + 3])
    data_4_y.append(data_y[i + 4])


class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


Batch_Size = 8
DataSet = DataSet(np.array(data_4_x), list(data_4_y))
train_size = int(len(data_4_x) * 0.8)
test_size = len(data_4_y) - train_size

# 划分训练集和测试集，并且将其转化为DataLoader
train_dataset, test_dataset = torch.utils.data.random_split(DataSet, [train_size, test_size])
# 使用切片操作分割数据
# train_dataset = DataSet[:train_size]
# test_dataset = DataSet[train_size:]
TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        chunk = x.chunk(x.size(-1), dim=2)
        out = torch.Tensor([]).to(x.device)
        for i in range(len(chunk)):
            out = torch.cat((out, chunk[i] + self.pe[:chunk[i].size(0), ...]), dim=2)
        return out


# 掩码机制
def transformer_generate_tgt_mask(length, device):
    # mask = torch.triu(torch.ones(length, length, device=device)) == 1
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


# Transformer结构
class Transformer(nn.Module):
    def __init__(self, n_encoder_inputs, n_decoder_inputs, Sequence_length, d_model=512, dropout=0.1, num_layer=8):
        """
        初始化
        :param n_encoder_inputs: 输入数据的特征维度
        :param n_decoder_inputs: 解码器输入的特征维度，其实等于编码器输出的特征维度
        :param Sequence_length:  transformer输入数据，序列的长度
        :param d_model: 词嵌入特征维度
        :param dropout:
        :param num_layer: Transformer块的个数
        """
        # 调用父类的构造函数
        super(Transformer, self).__init__()

        # 创建输入序列位置编码和目标序列位置编码的嵌入层
        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)

        # 创建Transformer编码器层和解码器层
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=dropout,
                                                         dim_feedforward=4 * d_model)

        # 创建Transformer编码器和解码器
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=8)

        # 创建输入和输出特征的线性投影层
        self.input_projection = torch.nn.Linear(n_encoder_inputs, d_model)
        self.output_projection = torch.nn.Linear(n_decoder_inputs, d_model)

        # 创建一个线性层用于最终的输出
        self.linear = torch.nn.Linear(d_model, 1)
        self.ziji_add_linear = torch.nn.Linear(Sequence_length, 1)

    def encode_in(self, src):
        # 对输入进行线性投影
        src_start = self.input_projection(src).permute(1, 0, 2)  # 将原始数据的维度重新排列，将原来的维度1移到维度0的位置，将维度0移到维度1的位置，维度2保持不变
        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)

        # 创建输入序列的位置编码
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)
        # 将位置编码添加到输入序列中，并进行编码
        src = src_start + pos_encoder
        src = self.encoder(src) + src_start
        return src

    def decode_out(self, tgt, memory):
        # 对输出进行线性投影
        tgt_start = self.output_projection(tgt).permute(1, 0, 2)
        out_sequence_len, batch_size = tgt_start.size(0), tgt_start.size(1)
        # 创建目标序列的位置编码
        pos_decoder = (
            torch.arange(0, out_sequence_len, device=tgt.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        tgt = tgt_start + pos_decoder
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size , seq_len , d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        '''
        :param src:（输入数据）的维度：(sequence_length, batch_size, n_encoder_inputs)，
                    其中sequence_length是输入序列的长度，batch_size是批次大小，n_encoder_inputs是输入特征的维度。
        :param target_in:（解码器输入）的维度：(sequence_length, batch_size, n_decoder_inputs)，
                    其中 sequence_length 是解码器输入序列的长度，batch_size 是批次大小，n_decoder_inputs 是解码器输入特征的维度。
        :return:
        '''
        src = self.encode_in(src)
        out = self.decode_out(tgt=target_in, memory=src)
        # 使用全连接变成[batch,1]构成基于transformer的回归单值预测
        out = out.squeeze(2)  # shape:[16,3,1]-->[16,3]
        out = self.ziji_add_linear(out)  # [16,3]-->[16,1]
        return out


model = Transformer(n_encoder_inputs=5, n_decoder_inputs=5, Sequence_length=3).to(device)


def test():
    with torch.no_grad():
        val_epoch_loss = []
        for index, (inputs, targets) in enumerate(TrainDataLoader):
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()
            tgt_in = torch.rand((Batch_Size, 3, 5))
            # tgt_in = inputs  # Use inputs as targets during testing
            outputs = model(inputs, inputs)
            loss = criterion(outputs.float(), targets.float())
            val_epoch_loss.append(loss.item())
    return np.mean(val_epoch_loss)



epochs = 50
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss().to(device)

# 训练模型
val_loss = []
train_loss = []
best_test_loss = 10000000  # 用于跟踪最佳验证损失，初始值设置为一个较大的数。
for epoch in tqdm(range(epochs)):
    train_epoch_loss = []
    for index, (inputs, targets) in enumerate(TrainDataLoader):
        inputs = torch.tensor(inputs).to(device)
        targets = torch.tensor(targets).to(device)
        inputs = inputs.float()
        targets = targets.float()

        tgt_in = torch.rand((Batch_Size, 3, 5))
        # tgt_in = inputs  # Use the same input as targets during training

        outputs = model(inputs, inputs)

        loss = criterion(outputs.float(), targets.float())
        # 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        train_epoch_loss.append(loss.item())
    train_loss.append(np.mean(train_epoch_loss))
    val_epoch_loss = test()
    val_loss.append(val_epoch_loss)
    # print("epoch:", epoch, "train_epoch_loss", train_epoch_loss, "val_epoch_loss:", val_epoch_loss)
    # np.savez('modelloss/loss.npz', y1=train_loss, y2=val_loss)
    # 保存下来最好的模型
    if val_epoch_loss < best_test_loss:
        best_test_loss = val_epoch_loss
        best_model = model
        print("best_test_loss ---------------------------", best_test_loss)
        torch.save(best_model.state_dict(), 'best_Transformer_trainModel.pth')

# 加载上一次的loss
# train_loss = np.load('modelloss/loss.npz')['y1'].reshape(-1, 1)
# val_loss = np.load('modelloss/loss.npz')['y2'].reshape(-1, 1)
# 画loss图
fig = plt.figure(facecolor='white', figsize=(10, 7))
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xmax=len(val_loss), xmin=0)
plt.ylim(ymax=max(max(train_loss), max(val_loss)), ymin=0)
# 画两条（0-9）的坐标轴并设置轴标签x ，y
x1 = [i for i in range(0, len(train_loss), 1)]  # 随机产生300个平均值为2，方差为1.2的浮点数，即第一簇点的x轴坐标
y1 = val_loss
x2 = [i for i in range(0, len(train_loss), 1)]
y2 = train_loss
area = np.pi * 4 ** 1
# 画散点图
plt.scatter(x1, y1, s=area, c='black', alpha=0.4, label='val_loss')
plt.scatter(x2, y2, s=area, c='red', alpha=0.4, label='train_loss')
plt.legend()
plt.show()

# 加载模型预测
model = Transformer(n_encoder_inputs=5, n_decoder_inputs=5, Sequence_length=3).to(device)
model.load_state_dict(torch.load('best_Transformer_trainModel.pth'))
model.to(device)
model.eval()
# 在对模型进行评估时，应该配合使用wit torch.nograd() 与 model.eval()

# 开始预测
y_pred = []
y_true = []
with torch.no_grad():
    val_epoch_loss = []
    for index, (inputs, targets) in enumerate(TestDataLoader):
        inputs = torch.tensor(inputs).to(device)
        targets = torch.tensor(targets).to(device)
        inputs = inputs.float()
        targets = targets.float()
        tgt_in = torch.rand((Batch_Size, 3, 5))
        # tgt_in = inputs  # Use the same input as targets during training
        outputs = model(inputs, inputs)
        outputs = list(outputs.cpu().numpy().reshape([1, -1])[0])
        targets = list(targets.cpu().numpy().reshape([1, -1])[0])
        y_pred.extend(outputs)
        y_true.extend(targets)

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr


# 画折线图
print("y_pred", y_pred)
print("y_true", y_true)
len_ = [i for i in range(len(y_pred))]

# plt.plot(len_, y_true, label='y_true', color='blue')
# plt.plot(len_, y_pred, label='y_pred', color='yellow')

plt.plot(len_, normalize_array(y_true), label='y_true', color='blue')
plt.plot(len_, normalize_array(y_pred), label='y_pred', color='yellow')
plt.show()
