"""
__author__ = 'Cheng Yuchao'
__project__: 大庆油田相关性分析 对比
__time__:  2024/4/16
__email__:"2477721334@qq.com"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入数据

data1 = pd.read_excel('../result/wlp_transformer/相关性分析lstm_BHC_全部特征_result.xlsx')
data2 = pd.read_excel('../result/wlp_transformer/相关性分析lstm_BHC_高相关性特征_result.xlsx')

data1_true = data1['lstm_BHC_true']
data1_pred = data1['lstm_BHC_predicted']
data2_pred = data2['lstm_BHC_predicted']
data2_true = data2['lstm_BHC_true']

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(data1_true, label='原测井曲线', color='black', linewidth=1)
ax.plot(data1_pred, label='所有特征作为输入', color='red', linewidth=1, linestyle='--')
ax.plot(data2_pred, label='高相关性特征作为输入', color='blue', linewidth=1, linestyle=':')
ax.legend()

# 设置横、纵坐标的标目
ax.set_xlabel('深度/m')
ax.set_ylabel('BHC/(g/cm^3)')

# 将横、纵坐标的刻度线置于坐标轴内侧
ax.tick_params(axis='both', direction='in')

# 显示图形
plt.show()
