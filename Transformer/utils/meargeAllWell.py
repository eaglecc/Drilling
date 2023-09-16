"""
__author__ = 'Cheng Yuchao'
__project__: 合并已有的.csv数据
__time__:  2023/09/14
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
data1 = pd.read_csv('../data/Well1_EPOR0_1.csv')
data2 = pd.read_csv('../data/Well2_EPOR0_1.csv')
data3 = pd.read_csv('../data/Well3_EPOR0_1.csv')
data4 = pd.read_csv('../data/Well4_EPOR0_1.csv')
data5 = pd.read_csv('../data/Well5_EPOR0_1.csv')
data6 = pd.read_csv('../data/Well6_EPOR0_1.csv')
data_x = data1[['DEPTH','GR', 'NPHI','DENSITY','PEF', 'VSHALE', 'DPHI', 'EPOR0','LITH']].values

Merge1 = pd.concat([data1,data2,data3,data4,data5,data6])

Merge1.to_csv('../data/Well_Merge_All_Data.csv',index=False)