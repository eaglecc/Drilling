"""
__author__ = 'Cheng Yuchao'
__project__: ARMA:自回归移动平均模型
__time__:  2023/09/30
__email__:"2477721334@qq.com"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import statsmodels.api as sm


warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查是否有可用的GPU，如果没有则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据
data = pd.read_csv('../../data/Well1_EPOR0_1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['NPHI', 'DENSITY', 'VSHALE', 'DPHI', 'EPOR0', 'LITH']]
data_y = data['GR']

# 归一化
min_value_y = data_y.min()  # 训练时y的最小值
max_value_y = data_y.max()  # 训练时y的最大值
data_x = (data_x - data_x.min()) / (data_x.max() - data_x.min())
data_y = (data_y - min_value_y) / (max_value_y - min_value_y)

# 2. Create a function to fit an ARMA model and make predictions:
def fit_arma_model(train_data, p, q):
    arma_model = sm.tsa.ARMA(train_data, order=(p, q))
    arma_result = arma_model.fit()
    return arma_result

def predict_arma_model(arma_result, test_data):
    arma_predictions = arma_result.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)
    return arma_predictions









