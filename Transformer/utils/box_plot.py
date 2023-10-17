"""
__author__ = 'Cheng Yuchao'
__project__: 绘制箱型图、替换离群点数据
__time__:  2023/09/27
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
data = pd.read_csv('../data/Well1_EPOR0_1.csv')
data_x = data[['GR', 'NPHI','DENSITY',  'VSHALE', 'DPHI', 'EPOR0']]
data_y = data['DEPTH']

# 绘制箱型图方法1
# for column in data_x.columns:
#     plt.figure(figsize=(6, 4))  # 可选：设置每个子图的尺寸
#     plt.boxplot(data_x[column], vert=False)  # 绘制箱型图，vert=False表示水平箱型图
#     plt.title(f'Box Plot of {column}')  # 设置图表标题
#     plt.xlabel(column)  # 设置x轴标签
#     plt.show()  # 显示图表


# 绘制箱型图方法2
sns.set(style="whitegrid")  # 设置Seaborn样式
# 遍历数据列并在子图中绘制箱型图
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10)) # 创建一个包含六个子图的图形
for i, column in enumerate(data_x.columns):
    ax = sns.boxplot(x=data_x[column], orient="h", ax=axes[i // 3, i % 3] ,whis=2.5)
    axes[i // 3, i % 3].set_title(f'Box Plot of {column}')
    # axes[i // 3, i % 3].set_xlabel(column)
    axes[i // 3, i % 3].set(xlabel=None)  # 清除x轴标签

plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()  # 显示图表

# 3. 查找离群点数据
outliers = {}  # 创建一个字典，用于存储离群点数据
for column in data_x.columns:
    boxplot = sns.boxplot(x=data_x[column], orient="h")
    outliers[column] = boxplot.get_lines()

# 4. 移动平均平滑序列数据
window_size = 10  # 滚动窗口的大小
res_data = pd.DataFrame()
# res_data['GR'] = data_x['GR'].rolling(window=window_size, min_periods=1).mean()
res_data['NPHI'] = data_x['NPHI'].rolling(window=window_size, min_periods=1).mean()
# res_data['DENSITY'] = data_x['DENSITY'].rolling(window=window_size, min_periods=1).mean()
# res_data['VSHALE'] = data_x['VSHALE'].rolling(window=window_size, min_periods=1).mean()
# res_data['DPHI'] = data_x['DPHI'].rolling(window=window_size, min_periods=1).mean()
res_data['EPOR0'] = data_x['EPOR0'].rolling(window=window_size, min_periods=1).mean()


i = 0
