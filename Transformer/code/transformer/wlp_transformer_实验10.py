"""
__author__ = 'Cheng Yuchao'
__project__:
原始Transformer对比实验
实验3: 缺失测井曲线预测实验：大庆油田井A数据集中  预测BHC
__time__:  2024/4/4
__email__:"2477721334@qq.com"
"""
import time
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
import causal_convolution_layer
from einops import rearrange
import os

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 导入数据
data = pd.read_csv('../../data/daqingyoutian/vertical_all_A1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['RMN-RMG','GR      .   ','SP      .mv  ', 'DEN     .g/cm3 ']].values
data_y = data['BHC     .'].values
input_features = 4

min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
#  Min-Max归一化
def min_max_normalize(data):
    min_vals = np.min(data, axis=0)  # 沿着列计算最小值
    max_vals = np.max(data, axis=0)  # 沿着列计算最大值
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data
data_x = min_max_normalize(data_x)
data_y = min_max_normalize(data_y)

# 2. 定义回看窗口大小
look_back = 500
future_window = 500

X, y = [], []
for i in range(len(data_x) - look_back + 1):
    X.append(data_x[i:i + look_back])
    y.append(data_y[i:i + look_back])
X = np.array(X)
y = np.array(y)

# 3. 划分数据集为训练集和测试集
train_size1 = int(0.4 * len(X))
train_size2 = int(0.6 * len(X))

train_features1 = X[:train_size1]
train_features2 = X[train_size2:]
train_features = np.concatenate((train_features1, train_features2), axis=0)

train_target1 = y[:train_size1]
train_target2 = y[train_size2:]
train_target = np.concatenate((train_target1, train_target2), axis=0)
test_features = X[train_size1:train_size2]
test_target = y[train_size1:train_size2]

train_features = torch.FloatTensor(train_features).to(device)
train_target = torch.FloatTensor(train_target).to(device)
test_features = torch.FloatTensor(test_features).to(device)


# 定义批次大小
batch_size = 1  # 可以根据需求调整

# 使用DataLoader创建数据加载器
train_dataset = torch.utils.data.TensorDataset(train_features, train_target)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


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


class Input_Embedding(nn.Module):
    def __init__(self, in_channels=1, res_num=4, feature_num=16):
        """
        Input_Embedding layer
        Args:
            in_channels: (int) input channels
            res_num: (int) resnet block number
            feature_num: (int) high-level feature number
        """
        super().__init__()
        # lstm
        self.lstm = nn.LSTM(input_features, feature_num, 1, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        # cnn + rescnn
        self.conv1 = nn.Sequential(
            # nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, padding=3, stride=1),
            nn.Conv1d(in_channels=in_channels, out_channels=feature_num, kernel_size=7, padding=3, stride=1),
            # nn.BatchNorm1d(feature_num),
            nn.ReLU()
        )
        self.rescnn = nn.ModuleList(
            [ResCNN(in_channels=feature_num, out_channels=feature_num, kernel_size=7, padding=3, stride=1) for i
             in range(res_num)])

    def forward(self, x):
        """
        conv1: [B,C,L]-->[B,feature_num,L]
        rescnn:[B,feature_num,L]-->[B,feature_num,L]
        Args:
            x:input sequence,shape[B,C,L] B:batch size，C：input channels,default=1，L:sequence len

        Returns:
            x:[B,feature_num,L]
        """
        x = self.conv1(x)  # conv1: [B,C,L]-->[B,feature_num,L]

        # lstm层
        # x = x.permute(0, 2, 1)
        # x, _ = self.lstm(x)  # （3050，50，6）-->（3050，50，16）
        # x = self.dropout(x)
        # x = x.permute(0, 2, 1)

        # for model in self.rescnn:  # rescnn:[B,feature_num,L]-->[B,feature_num,L]
        #     x = model(x)
        return x


class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        """
        Args:
            in_channels: (int)
            out_channels: (int)
            kernel_size: (int)
            padding: (int)
            stride: (int)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                      stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: [B,C,L]

        Returns:

        """
        identity = x
        out = self.conv(x)
        return identity + out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
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


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout, seq_len):
#         """
#         Position Embedding
#         Args:
#             d_model: (int) feature dimension
#             dropout: (float) drop rate
#             seq_len: (int) sequence length
#         """
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(seq_len, d_model).float()
#         position = torch.arange(0, seq_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).float()
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         """
#
#         Args:
#             x:[B,L,C] batch_size,sequence len,channel number
#
#         Returns:
#
#         """
#         x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
#                                         requires_grad=False)
#         return self.dropout(x)
#

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """

        Args:
            dim: (int) feature dimension
            num_heads: (int) mutil heads number
            mlp_ratio: (float) hidden layer ratio in feedforward
            qkv_bias: (bool) use bias or not in qkv linear
            drop: (float) feedforward dropout
            attn_drop: (float) attention dropout
            act_layer: activation layer default (nn.GELU)
            norm_layer: nomrmalization layer default (nn.LayerNorm)
        """
        super().__init__()
        self.norm1 = norm_layer(dim)

        # 普通注意力：SelfAttention
        self.attn = SelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # Vector: Attention_Rel_Vec
        # self.attn = Attention_Rel_Vec(emb_size=dim, num_heads=num_heads, seq_len=44, dropout=attn_drop)
        # eRPE: Attention_Rel_Scl
        # self.attn = Attention_Rel_Scl(emb_size=dim, num_heads=8, seq_len=80, dropout=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.feedforward = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # 注意力
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.feedforward(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, block_num, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.block = nn.ModuleList(
            [TransformerEncoder(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                attn_drop=attn_drop, act_layer=act_layer, norm_layer=norm_layer) for i in
             range(block_num)])

    def forward(self, x):
        for model in self.block:
            x = model(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """

        Args:
            dim: (int) feature dimension
            num_heads: (int) mutil heads number
            qkv_bias:  (bool) use bias or not in qkv linear
            attn_drop: (float) attention drop rate
            proj_drop: (float) linear drop rate
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """

        Args:
            x: [B,L,C] batch_size,sequence len,channel number

        Returns:

        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_Rel_Vec(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.Er = nn.Parameter(torch.randn(self.seq_len, int(emb_size / num_heads)))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(self.seq_len, self.seq_len))
            .unsqueeze(0).unsqueeze(0)
        )

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        QEr = torch.matmul(q, self.Er.transpose(0, 1))
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, self.num_heads, seq_len, seq_len)

        attn = torch.matmul(q, k)
        # attn shape (seq_len, seq_len)
        attn = (attn + Srel) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = nn.functional.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """

        Args:
            in_features: (int) input feature number
            hidden_features: (int) hidden_features number
            out_features: (int) output feature number
            act_layer: activation function
            drop: drop rate
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, res_num=4, out_channels=1, feature_num=16):
        """

        Args:
            out_channels: (int) output feature number
            res_num: (int) resnet block number
            feature_num: (int) feature dimension
        """
        super().__init__()
        self.rescnn = nn.ModuleList(
            [ResCNN(in_channels=feature_num, out_channels=feature_num, kernel_size=7, padding=3, stride=1) for
             i in range(res_num)])
        self.out_layer = nn.Sequential(
            nn.Conv1d(in_channels=feature_num, out_channels=out_channels, kernel_size=7, padding=3, stride=1),
            nn.Sigmoid()  # normal
            # nn.ReLU()   # nonormal
        )

    def forward(self, x):
        """
        Args:
            x: input sequence,shape[B,C,L] B:batch size，C：特征维度,default=1，L序列长度

        Returns:
        """
        # for model in self.rescnn:
        #     x = model(x)
        x = self.out_layer(x)
        # x = x[:, :, -1] # 取最后一个时间步的输出
        return x


# Transformer结构
class Transformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_num=16, res_num=4, encoder_num=4, use_pe=False,
                 dim=16, seq_len=500, num_heads=2, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 position_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm
                 ):
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
        self.use_pe = use_pe
        # 位置编码层
        self.input_positional_encoding = PositionalEncoding(dim, max_len=look_back)
        self.target_positional_encoding = PositionalEncoding(dim, max_len=look_back)

        # 创建输入序列位置编码和目标序列位置编码的嵌入层
        self.input_pos_embedding = torch.nn.Embedding(500, embedding_dim=dim)
        self.target_pos_embedding = torch.nn.Embedding(500, embedding_dim=dim)
        # 创建输入和输出特征的线性投影层
        self.input_projection = torch.nn.Linear(input_features, dim)
        self.output_projection = torch.nn.Linear(input_features, dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=drop,
                                                         dim_feedforward=2 * dim)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dropout=drop,
                                                         dim_feedforward=2 * dim)

        # 创建Transformer编码器和解码器
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.linear = torch.nn.Linear(dim, 1)
        self.fc = nn.Linear(look_back, future_window)

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
        # src = positional_encoding + pos_encoder + src_start
        src = positional_encoding + pos_encoder
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
        # 对输入数据嵌入编码
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)
        # 位置信息编码
        positional_encoding = self.input_positional_encoding(tgt_start)
        tgt = positional_encoding + pos_decoder
        # tgt = positional_encoding + pos_decoder + tgt_start
        # 掩码
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        # 送到解码器模型中
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask) + tgt_start
        out = self.linear(out)
        out = out.permute(1, 2, 0)  # [batch_size , seq_len , d_model]
        out = self.fc(out)
        return out

    def forward(self, x):
        target_in = x
        # encoder
        src = self.encode_in(x)
        # decoder
        out = self.decode_out(tgt=target_in, memory=src)
        # x = self.decoder(x)
        return out


model = Transformer(in_channels=input_features, out_channels=1, feature_num=16).to(device)

# 计算参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params = count_parameters(model)
num_params_in_k = num_params / 1000
num_params_in_million = num_params / 1000000

print(f"Number of parameters in the Transformer model: {num_params}")
print(f"Number of parameters in the Transformer model: {num_params_in_k:.2f}K")
print(f"Number of parameters in the Transformer model: {num_params_in_million:.3f}M")


def test():
    with torch.no_grad():
        val_epoch_loss = []
        for inputs, targets in train_loader:
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            loss = criterion(outputs.float(), targets.float())
            val_epoch_loss.append(loss.item())
    return np.mean(val_epoch_loss)


epochs = 20
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss().to(device)

# 训练模型
train_model = True
if train_model:
    val_loss = []
    train_loss = []
    best_test_loss = 10000000  # 用于跟踪最佳验证损失，初始值设置为一个较大的数。
    total_training_time = 0.0  # 初始化总的训练时间

    for epoch in tqdm(range(epochs)):
        train_epoch_loss = []
        start_time = time.time()  # 记录每个 epoch 的开始时间

        for inputs, targets in train_loader:
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)
            outputs = outputs.squeeze(dim=0)
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

        # 计算并打印每个 epoch 的训练时间
        end_time = time.time()
        epoch_time = end_time - start_time
        total_training_time += epoch_time
        print(f"Epoch {epoch + 1}/{epochs}, OETT: {epoch_time:.2f} seconds")

        # 保存下来最好的模型
        if val_epoch_loss < best_test_loss:
            best_test_loss = val_epoch_loss
            best_model = model
            print("best_test_loss ---------------------------", best_test_loss)
            torch.save(best_model.state_dict(), '../pth/best_Transformer_trainModel_BHC_缺失.pth')

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
    area = np.pi * 5 ** 1
    # 画散点图
    plt.scatter(x1, y1, s=area, c='black', alpha=0.4, label='val_loss')
    plt.scatter(x2, y2, s=area, c='red', alpha=0.4, label='train_loss')
    plt.legend()
    plt.show()

# 加载模型预测
model = Transformer(in_channels=input_features, out_channels=1, feature_num=16).to(device)
model.load_state_dict(torch.load('../pth/best_Transformer_trainModel_BHC_缺失.pth'))
model.to(device)
model.eval()
# 在对模型进行评估时，应该配合使用wit torch.nograd() 与 model.eval()

# 8. 测试集预测
with torch.no_grad():
    predicted = model(test_features)
predicted = predicted.cpu().numpy()
# predicted1 = predicted[0, :, :].reshape(-1, 1)
# predicted2 = predicted[:, :, -1]
# predicted = np.concatenate((predicted1, predicted2))

predicted_train = predicted[:, :, 0]

pred = predicted[0, :, :].reshape(-1 , 1)   # 只取单行预测结果

# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
# test_target_train = test_target[:, 0]
# test_target_future = test_target[-1, :]
# test_target = np.concatenate((test_target_train, test_target_future))

test_target_train = test_target[:, 0]
test2 = test_target[0, :]

plt.plot(test_target_train, label='True')
plt.plot(predicted_train, label='Predicted')
plt.title('Transformer测井曲线预测')
plt.legend()
# 使用savefig保存图表为文件
plt.savefig(('../../result/transformer/T_experiment9_batch{}_epoch_{}.png').format(batch_size, epochs))  # 保存为PNG格式的文件
plt.show()


# 11. 存储预测结果
# 反归一化
predicted_original_data = predicted_train * (max_value_y - min_value_y) + min_value_y
test_target_original_data = test_target_train * (max_value_y - min_value_y) + min_value_y
file_name = '../../result/wlp_transformer/daqingyoutian_result缺失_BHC_原始T.xlsx'
# 如果文件不存在，创建一个新 Excel 文件并存储数据

df = pd.DataFrame()  # 创建一个新 DataFrame
df['well1_BHC_true'] = test_target_original_data
df['well1_BHC_predicted'] = predicted_original_data.flatten()
df.to_excel(file_name, index=False)  # index=False 防止写入索引列

