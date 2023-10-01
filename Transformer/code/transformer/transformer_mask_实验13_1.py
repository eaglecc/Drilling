"""
__author__ = 'Cheng Yuchao'
__project__: 实验13：集成学习实验：做3个岩性模型，分别调用
__time__:  2023/09/20
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
data = pd.read_csv('../../data/Well_Merge_All_Data.csv')
# data.dropna(axis=0, how='any')  #只要行中包含任何一个缺失值，就删除整行。
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['GR', 'NPHI', 'VSHALE', 'DPHI', 'EPOR0','WellNum', 'LITH']].values
data_y = data['DENSITY'].values

# 获取所有不同的LITH值
unique_lith_values = np.unique(data_x[:, -1])
# 创建一个字典，用于存储不同LITH值对应的数组
lith_arrays = {}
lith_targets = {}

# 根据每个不同的LITH值分割数据
for lith_value in unique_lith_values:
    lith_condition = data_x[:, -1] == lith_value
    lith_arrays[lith_value] = data_x[lith_condition,]
    lith_targets[lith_value] = data_y[lith_condition,]

# 根据字典大小训练各自对应的模型
lith_1_train = lith_arrays.get(1)
lith_2_train = lith_arrays.get(2)
lith_3_train = lith_arrays.get(3)
lith_1_targets = lith_targets.get(1)
lith_2_targets = lith_targets.get(2)
lith_3_targets = lith_targets.get(3)


# 使用对应岩性的数据集替换原数据(手动替换3次，得到三个训练模型)
data_x = lith_1_train
data_y = lith_1_targets


#  Min-Max归一化
min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

data_4_x = []
data_4_y = []

for i in range(0, len(data_y) - 44, 44):
    data_4_x.append(data_x[i:i + 44])
    data_4_y.append(data_y[i:i + 44])


class DataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.FloatTensor(data_inputs)
        self.label = torch.FloatTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)


Batch_Size = 1
DataSet = DataSet(np.array(data_4_x), list(data_4_y))
train_size = int(len(data_4_x) * 0.9)
test_size = len(data_4_y) - train_size

# 划分训练集和测试集
train_dataset = torch.utils.data.Subset(DataSet, list(range(train_size)))  # 训练集包含数据集的前 train_size 个数据
test_dataset = torch.utils.data.Subset(DataSet, list(range(train_size, len(data_4_x))))  # 测试集包含后 test_size 个数据
TrainDataLoader = Data.DataLoader(train_dataset, batch_size=Batch_Size, shuffle=False,drop_last=True)  # shuffle=False:不打乱顺序
TestDataLoader = Data.DataLoader(test_dataset, batch_size=Batch_Size, shuffle=False, drop_last=True)


# 掩码机制
def transformer_generate_tgt_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class Input_Embedding(nn.Module):
    def __init__(self, in_channels=1, res_num=4, feature_num=64):
        """
        Input_Embedding layer
        Args:
            in_channels: (int) input channels
            res_num: (int) resnet block number
            feature_num: (int) high-level feature number
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=feature_num, kernel_size=11, padding=5, stride=1),
            nn.BatchNorm1d(feature_num),
            nn.ReLU())
        self.rescnn = nn.ModuleList(
            [ResCNN(in_channels=feature_num, out_channels=feature_num, kernel_size=11, padding=5, stride=1) for i
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
        # conv1: [B,C,L]-->[B,feature_num,L]
        x = self.conv1(x)
        # rescnn:[B,feature_num,L]-->[B,feature_num,L]
        for model in self.rescnn:
            x = model(x)
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


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, d_model))  # 可学习的参数
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        # x = x + self.pe
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class Attention_Rel_Scl(nn.Module):  # Equation 14 page 12
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
        relative_coords = relative_coords.permute(1, 2, 0)  #c h w -> h w c
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
        # relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        relative_bias = relative_bias.permute(1, 0).unsqueeze(0).unsqueeze(2).unsqueeze(3)
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

        self.Er = nn.Parameter(torch.randn(self.seq_len, int(emb_size/num_heads)))

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

class tAPE(nn.Module):  # Equation 13 page 11
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin((position * div_term) * (d_model / max_len))
        pe[:, 1::2] = torch.cos((position * div_term) * (d_model / max_len))
        pe = scale_factor * pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        # x = x + self.pe
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


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
        # self.attn = SelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # self.attn = Attention_Rel_Scl(emb_size= dim, num_heads= num_heads, seq_len= 44,dropout= attn_drop)
        self.attn = Attention_Rel_Vec(emb_size= dim, num_heads= num_heads, seq_len= 44,dropout= attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.feedforward = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
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
    def __init__(self, res_num=4, out_channels=1, feature_num=64):
        """

        Args:
            out_channels: (int) output feature number
            res_num: (int) resnet block number
            feature_num: (int) feature dimension
        """
        super().__init__()
        self.rescnn = nn.ModuleList(
            [ResCNN(in_channels=feature_num, out_channels=feature_num, kernel_size=11, padding=5, stride=1) for
             i in range(res_num)])
        self.out_layer = nn.Sequential(
            nn.Conv1d(in_channels=feature_num, out_channels=out_channels, kernel_size=11, padding=5, stride=1),
            nn.Sigmoid()  # normal
            # nn.ReLU()   # nonormal
        )

    def forward(self, x):
        """
        Args:
            x: input sequence,shape[B,C,L] B:batch size，C：特征维度,default=1，L序列长度

        Returns:
        """
        for model in self.rescnn:
            x = model(x)
        x = self.out_layer(x)
        return x


# Transformer结构
class Transformer(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_num=64, res_num=4, encoder_num=4, use_pe=False,
                 dim=64, seq_len=160, num_heads=4, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
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
        self.tAPE_position_embedding = tAPE(d_model=dim, dropout=position_drop, max_len=seq_len)
        self.learnable_position_embedding = LearnablePositionalEncoding(d_model=dim, dropout=position_drop,
                                                                        max_len=seq_len)
        self.use_pe = use_pe
        self.transformer_encdoer = TransformerBlock(block_num=res_num, dim=dim, num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                                                    attn_drop=attn_drop,
                                                    act_layer=act_layer, norm_layer=norm_layer)
        self.decoder = Decoder(out_channels=out_channels, feature_num=feature_num, res_num=res_num)

        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=dim, nhead=4, dropout=drop,
                                                         dim_feedforward=2 * dim)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.linear = torch.nn.Linear(dim, 1)

    def decode_out(self, tgt, memory):
        # 掩码
        # tgt_mask = transformer_generate_tgt_mask(out_sequence_len, tgt.device)
        # 送到解码器模型中
        out = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=None)
        out = self.linear(out)
        return out

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_embedding(x)
        # [B,feature_num,L]--> [B,L,feature_num]
        x = x.transpose(-2, -1)
        if self.use_pe:
            # x = self.position_embedding(x)
            # x = self.tAPE_position_embedding(x)
            x = self.learnable_position_embedding(x)
        tgt_in = x
        # [B, L, feature_num] --> [B, L, feature_num]
        x = self.transformer_encdoer(x)
        out = self.decode_out(tgt=tgt_in, memory=x)
        # [B,  L, feature_num] --> [B, feature_num,L]
        out = out.transpose(-2, -1)
        # [[B, feature_num,L] --> [B,out_channels,L]
        # x = self.decoder(x)

        return out


model = Transformer(in_channels=7, out_channels=1, feature_num=64).to(device)


def test():
    with torch.no_grad():
        val_epoch_loss = []
        for index, (inputs, targets) in enumerate(TrainDataLoader):
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs)
            loss = criterion(outputs.float(), targets.float())
            val_epoch_loss.append(loss.item())
    return np.mean(val_epoch_loss)


epochs = 30
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
        for index, (inputs, targets) in enumerate(TrainDataLoader):
            inputs = torch.tensor(inputs).to(device)
            targets = torch.tensor(targets).to(device)
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)

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
            torch.save(best_model.state_dict(), './pth/best_Transformer_trainModel_LITH1.pth')

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
model = Transformer(in_channels=7, out_channels=1, feature_num=64).to(device)
model.load_state_dict(torch.load('./pth/best_Transformer_trainModel_LITH1.pth'))
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
        outputs = model(inputs)
        outputs = list(outputs.cpu().numpy().reshape([1, -1])[0])
        targets = list(targets.cpu().numpy().reshape([1, -1])[0])
        y_pred.extend(outputs)
        y_true.extend(targets)

# 画折线图
print("y_pred", y_pred)
print("y_true", y_true)
len_ = [i for i in range(len(y_pred))]

plt.plot(len_, y_true, label='y_true', color='blue')
plt.plot(len_, y_pred, label='y_pred', color='yellow')
plt.legend()
plt.show()

# 将列表转换为NumPy数组
array1 = np.array(y_true)
array2 = np.array(y_pred)

mse = ((array1 - array2) ** 2).mean()
mae = np.mean(np.abs(array2 - array1))
print("平均绝对误差（MAE）：", mae)
print("均方误差（MSE）：", mse)
