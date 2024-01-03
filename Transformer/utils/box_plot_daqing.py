"""
__author__ = 'Cheng Yuchao'
__project__: 绘制箱型图、替换离群点数据  大庆油田数据
__time__:  2023/11/15
__email__:"2477721334@qq.com"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 导入数据
data = pd.read_csv('../data/daqingyoutian/vertical_all_A1.csv')
data_x = data[['RMN-RMG', 'CAL     .cm ', 'SP      .mv  ', 'GR      .   ', 'DEN     .g/cm3 ', 'HAC     .us/m']]
# 创建新的列名列表
new_columns = ['RMN-RMG', 'CAL', 'SP', 'GR', 'DEN', 'HAC']
data_x.columns = new_columns
# data_x = data[['RMN-RMG', 'CAL     .cm ', 'SP      .mv  ', 'GR      .   ', 'DEN     .g/cm3 ', 'HAC     .us/m', 'BHC     .']]
data_y = data['DEPT    .M ']

# 绘制箱型图方法1
# for column in data_x.columns:
#     plt.figure(figsize=(6, 4))  # 可选：设置每个子图的尺寸
#     plt.boxplot(data_x[column], vert=False)  # 绘制箱型图，vert=False表示水平箱型图
#     plt.title(f'Box Plot of {column}')  # 设置图表标题
#     plt.xlabel(column)  # 设置x轴标签
#     plt.show()  # 显示图表


# 绘制箱型图方法2
# sns.set(style="darkgrid")
# sns.set(style="dark")
# sns.set(style="white")
# sns.set(style="ticks")
sns.set(style="whitegrid")  # 设置Seaborn样式
sns.boxplot(data=data_x,palette="Set3")
# fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(15, 4))  # 一行六列，调整图形大小
#
# for i, column in enumerate(data_x.columns):
#     sns.boxplot(x=data_x[column], ax=axes[i])
#     axes[i].set_title(f'Box Plot of {column}')
#
# plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()

# 3. 查找离群点数据
outliers = {}  # 创建一个字典，用于存储离群点数据
for column in data_x.columns:
    boxplot = sns.boxplot(x=data_x[column], orient="h")
    outliers[column] = boxplot.get_lines()

# 4. 移动平均平滑序列数据
window_size = 10  # 滚动窗口的大小
res_data = pd.DataFrame()
res_data['RMN-RMG'] = data_x['RMN-RMG'].rolling(window=window_size, min_periods=1).mean()

