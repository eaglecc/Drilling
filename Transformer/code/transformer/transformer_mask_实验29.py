"""
__author__ = 'Cheng Yuchao'
__project__:
原始Transformer+ Input Embedding(CNN) 对比实验
实验28: 测井曲线补全实验：井C数据集中使用伽马射线值、中子孔隙度、光电因子、有效孔隙度  预测  密度
__time__:  2023/10/19
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
import causal_convolution_layer
from einops import rearrange
import os

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 导入数据
data = pd.read_csv('../../data/Well3_EPOR0_1.csv')
# data.dropna(axis=0, how='any')  #只要行中包含任何一个缺失值，就删除整行。
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
# data_x = data[['DENSITY', 'NPHI', 'VSHALE', 'DPHI', 'EPOR0', 'LITH']].values
data_x = data[['GR', 'NPHI', 'PEF', 'EPOR0']]
# 异常值处理：NPHI和PEF不为负值
data_x.loc[data_x['NPHI'] < 0, 'NPHI'] = 0
data_x.loc[data_x['PEF'] < 0, 'PEF'] = 0
window_size = 6  # 移动平均窗口大小
data_x['PEF'] = data_x['PEF'].rolling(window=window_size).mean()
data_x = data_x.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data_x.values

data_y = data['DENSITY'].values
input_features = 4

#  Min-Max归一化
min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 2. 定义回看窗口大小
look_back = 80
# 创建回看窗口数据
X, y = [], []
for i in range(len(data_x) - look_back):
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
    def __init__(self, in_channels=1, res_num=4, feature_num=128):
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
        # x, _ = self.lstm(x)  # （3050，50，6）-->（3050，50，64）
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
    def __init__(self, d_model, dropout, seq_len):
        """
        Position Embedding
        Args:
            d_model: (int) feature dimension
            dropout: (float) drop rate
            seq_len: (int) sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(seq_len, d_model).float()
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).float()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        Args:
            x:[B,L,C] batch_size,sequence len,channel number

        Returns:

        """
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)],
                                        requires_grad=False)
        return self.dropout(x)


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
    def __init__(self, res_num=4, out_channels=1, feature_num=128):
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
    def __init__(self, in_channels=1, out_channels=1, feature_num=128, res_num=4, encoder_num=4, use_pe=False,
                 dim=128, seq_len=160, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
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
        self.feature_embedding = Input_Embedding(in_channels=in_channels, feature_num=feature_num, res_num=res_num)
        self.position_embedding = PositionalEncoding(d_model=dim, dropout=position_drop, seq_len=seq_len)
        self.use_pe = use_pe
        self.transformer_encdoer = TransformerBlock(block_num=4, dim=dim, num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                                    attn_drop=attn_drop,
                                                    act_layer=act_layer, norm_layer=norm_layer)
        # self.decoder = Decoder(out_channels=out_channels, feature_num=feature_num, res_num=res_num)

        # 创建输入序列位置编码和目标序列位置编码的嵌入层
        self.input_pos_embedding = torch.nn.Embedding(5000, embedding_dim=dim)
        self.target_pos_embedding = torch.nn.Embedding(5000, embedding_dim=dim)
        # 创建输入和输出特征的线性投影层
        self.input_projection = torch.nn.Linear(input_features, dim)
        self.output_projection = torch.nn.Linear(input_features, dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=drop,
                                                         dim_feedforward=2 * dim)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, dropout=drop,
                                                         dim_feedforward=2 * dim)

        # 创建Transformer编码器和解码器
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.linear = torch.nn.Linear(dim, 1)

        # 使用Local Attention：即causal_convolution_layer
        self.causal_input_embedding = causal_convolution_layer.context_embedding(6, feature_num, 9)

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
        positional_encoding = self.position_embedding(src_start)
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
        positional_encoding = self.position_embedding(tgt_start)
        tgt = positional_encoding + pos_decoder
        # tgt = positional_encoding + pos_decoder + tgt_start
        # 掩码
        tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        # 送到解码器模型中
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=None) + tgt_start
        out = out.permute(1, 0, 2)  # [batch_size , seq_len , d_model]
        out = self.linear(out)
        return out

    def forward(self, x):
        target_in = x
        # x = x.permute(0, 2, 1)
        # x = self.feature_embedding(x)
        # # [B,feature_num,L]--> [B,L,feature_num]
        # x = x.transpose(-2, -1)
        # if self.use_pe:
        #     x = self.position_embedding(x)
        # x = self.transformer_encdoer(x)
        # x = x.permute(1, 0, 2)

        # encoder
        src = self.encode_in(x)
        # decoder
        out = self.decode_out(tgt=target_in, memory=src)
        # x = self.decoder(x)
        return out


model = Transformer(in_channels=input_features, out_channels=1, feature_num=128).to(device)


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


epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.MSELoss().to(device)

# 训练模型
train_model = True
if train_model:

    val_loss = []
    train_loss = []
    best_test_loss = 10000000  # 用于跟踪最佳验证损失，初始值设置为一个较大的数。
    for epoch in tqdm(range(epochs)):
        train_epoch_loss = []
        for inputs, targets in train_loader:
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 1)
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
        # 保存下来最好的模型
        if val_epoch_loss < best_test_loss:
            best_test_loss = val_epoch_loss
            best_model = model
            print("best_test_loss ---------------------------", best_test_loss)
            torch.save(best_model.state_dict(), '../pth/best_Transformer_trainModel28_DENSITY.pth')

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
model = Transformer(in_channels=input_features, out_channels=1, feature_num=128).to(device)
model.load_state_dict(torch.load('../pth/best_Transformer_trainModel28_DENSITY.pth'))
model.to(device)
model.eval()
# 在对模型进行评估时，应该配合使用wit torch.nograd() 与 model.eval()

# 8. 测试集预测
with torch.no_grad():
    predicted = model(torch.FloatTensor(X).to(device))
predicted = predicted.cpu().numpy()
predicted = predicted[:, :, -1]
# 9. 绘制真实数据和预测数据的曲线
plt.figure(figsize=(12, 6))
test_target = y[:, -1]
plt.plot(test_target, label='True')
plt.plot(predicted, label='Predicted')
plt.title('Transformer测井曲线预测')
plt.legend()
# plt.xlim(1900, 2900)
# 使用savefig保存图表为文件
plt.savefig(('../../result/transformer/experiment28_batch{}_epoch_{}.png').format(batch_size, epochs))  # 保存为PNG格式的文件
plt.show()

# 10. Calculate RMSE、MAPE
mse = np.mean((test_target - predicted) ** 2)
rmse = np.sqrt(np.mean((test_target - predicted) ** 2))
mae = np.mean(np.abs(test_target - predicted))
mape = np.mean(np.abs((test_target - predicted) / test_target))
print("MSE", mse)  # 0.09753167668446872
print("RMSE", rmse)  # 0.3123006190907548
print("MAPE:", mape)  # 180.2477638456806 %
# 创建一个txt文件并将结果写入其中
resultpath = ('../../result/transformer/experiment28_batch{}_epoch_{}.txt').format(batch_size, epochs)
with open(resultpath, "w") as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"MAPE: {mape}\n")

print("预测结果已写入experiment15_epoch_{}.txt文件")

# 11. 存储预测结果
file_name = '../../result/transformer_result.xlsx'
# 如果文件不存在，创建一个新 Excel 文件并存储数据
df = pd.DataFrame({'well3_DEN_predicted': predicted.flatten()})  # 创建一个新 DataFrame
df['well3_DEN_true'] = test_target
df.to_excel(file_name, index=False)  # index=False 防止写入索引列
