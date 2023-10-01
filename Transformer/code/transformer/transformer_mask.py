"""
__author__ = 'Cheng Yuchao'
__project__: 实验5-2: 根据预测集所占岩性的种类判断用那个模型
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
data = pd.read_csv('../../data/Well4_EPOR0_1.csv')
# data.dropna(axis=0, how='any')  #只要行中包含任何一个缺失值，就删除整行。
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['GR', 'NPHI', 'VSHALE', 'DPHI', 'EPOR0', 'LITH']].values
data_y = data['DENSITY'].values

# 获取所有不同的LITH值
unique_lith_values = np.unique(data_x[:, -1])
# 创建一个字典，用于存储不同LITH值对应的数组
lith_arrays = {}
lith_targets = {}


# 四个数据划分为一组，用前三个数据预测后一个
data_5_x = []
data_5_y = []

for i in range(0, len(data_y) - 5, 1):
    # data_5_x.append(data_x[i:i + 3])
    # data_5_y.append(data_y[i + 5])
    data_5_x.append(data_x[[i, i + 1, i + 3, i + 4], :])
    data_5_y.append(data_y[i + 2])


class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


Batch_Size = 32
DataSet = DataSet(np.array(data_5_x), list(data_5_y))
train_size = int(len(data_5_x) * 0.8)
test_size = len(data_5_y) - train_size


# 根据测试集中岩性种类判断选用那个模型预测
test_lith = data_x[-test_size:]
last_column = test_lith[:,-1]
#    使用 NumPy 统计函数计算数字 1、2、3 的出现次数
flat_data = last_column.flatten()
count_0 = np.count_nonzero(flat_data == 0)
count_1 = np.count_nonzero(flat_data == 1)
count_2 = np.count_nonzero(flat_data == 2)
#    计算各自的占比
total_count = len(flat_data)
percentage_0 = (count_0 / total_count) * 100
percentage_1 = (count_1 / total_count) * 100
percentage_2 = (count_2 / total_count) * 100
max_percentage = max(percentage_0, percentage_1, percentage_2)


# 划分训练集和测试集
train_dataset = torch.utils.data.Subset(DataSet, list(range(train_size)))  # 训练集包含数据集的前 train_size 个数据
test_dataset = torch.utils.data.Subset(DataSet, list(range(train_size, len(data_5_x))))  # 测试集包含后 test_size 个数据

TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=False,
                                  drop_last=True)  # shuffle=False:不打乱顺序
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        ## 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面这个代码只是其中一种实现方式；
        ## 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        ## pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        ##假设我的demodel是512，2i那个符号中i从0取到了255，那么2i对应取值就是0,2,4...510
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  ## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  ##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        ## 上面代码获取之后得到的pe:[max_len*d_model]
        ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

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
        self.input_positional_encoding = PositionalEncoding(d_model, max_len=Sequence_length)
        self.target_positional_encoding = PositionalEncoding(d_model, max_len=Sequence_length)

        # 创建输入序列位置编码和目标序列位置编码的嵌入层
        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=d_model)

        # 创建Transformer编码器层和解码器层
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dropout=dropout,
                                                         dim_feedforward=5 * d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dropout=dropout,
                                                         dim_feedforward=5 * d_model)

        # 创建Transformer编码器和解码器
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)

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
        # 对输入数据编码
        embedding_encoder = self.input_pos_embedding(pos_encoder)
        pos_encoder = embedding_encoder.permute(1, 0, 2)
        # 位置信息编码
        positional_encoding = self.input_positional_encoding(src_start)
        # 将位置编码添加到输入序列中，并输入编码器中
        src = positional_encoding + pos_encoder + src_start
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
        # 对输入数据编码
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        # 位置信息编码
        positional_encoding = self.input_positional_encoding(tgt_start)
        tgt = positional_encoding + pos_decoder + tgt_start
        # 掩码
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        # 送到解码器模型中
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=None) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size , seq_len , d_model]
        out = self.linear(out)
        return out

    def forward(self, src, target_in):
        '''
        :param src:（输入数据）的维度：(batch_size,sequence_length,  n_encoder_inputs)，
                    其中sequence_length是输入序列的长度，batch_size是批次大小，n_encoder_inputs是输入特征的维度。
        :param target_in:（解码器输入）的维度：( batch_size,sequence_length, n_decoder_inputs)，
                    其中 sequence_length 是解码器输入序列的长度，batch_size 是批次大小，n_decoder_inputs 是解码器输入特征的维度。
        :return:
        '''
        src = self.encode_in(src)
        out = self.decode_out(tgt=target_in, memory=src)
        # 使用全连接变成[batch,1]构成基于transformer的回归单值预测
        out = out.squeeze(2)  # shape:[16,3,1]-->[16,3]
        out = self.ziji_add_linear(out)  # [16,3]-->[16,1]
        return out


# 加载模型预测
if max_percentage == percentage_0:
    print("岩性0占比最大：", max_percentage)
    model = Transformer(n_encoder_inputs=6, n_decoder_inputs=6, Sequence_length=4).to(device)
    model.load_state_dict(torch.load('Transformer_trainModel_LITH0.pth'))
elif max_percentage == percentage_1:
    print("岩性1占比最大：", max_percentage)
    model = Transformer(n_encoder_inputs=6, n_decoder_inputs=6, Sequence_length=4).to(device)
    model.load_state_dict(torch.load('Transformer_trainModel_LITH1.pth'))
else:
    model = Transformer(n_encoder_inputs=6, n_decoder_inputs=6, Sequence_length=4).to(device)
    model.load_state_dict(torch.load('Transformer_trainModel_LITH2.pth'))
    print("岩性2占比最大：", max_percentage)

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

plt.plot(len_, y_true, label='y_true', color='blue')
plt.plot(len_, y_pred, label='y_pred', color='yellow')
# plt.plot(len_, normalize_array(y_true), label='y_true', color='blue')
# plt.plot(len_, normalize_array(y_pred), label='y_pred', color='yellow')

plt.legend()
plt.show()

# 将列表转换为NumPy数组
array1 = np.array(y_true)
array2 = np.array(y_pred)

mse = ((array1 - array2) ** 2).mean()
mae = np.mean(np.abs(array2 - array1))
print("平均绝对误差（MAE）：", mae)
print("均方误差（MSE）：", mse)
