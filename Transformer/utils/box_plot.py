"""
__author__ = 'Cheng Yuchao'
__project__: 绘制箱型图
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

# 导入数据
data = pd.read_csv('../data/Well1_EPOR0_1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['GR', 'NPHI','DENSITY',  'VSHALE', 'DPHI', 'EPOR0','LITH']]
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
for column in data_x.columns:
    plt.figure(figsize=(8, 6))  # 可选：设置每个子图的尺寸
    sns.boxplot(x=data_x[column], orient="h")  # 绘制水平箱型图
    plt.title(f'Box Plot of {column}')  # 设置图表标题
    plt.xlabel(column)  # 设置x轴标签
    plt.show()  # 显示图表

# 绘制箱型图
# plt.figure(figsize=(12, 6))  # 可选：设置图表尺寸
# sns.boxplot(data=data_x, orient="v")  # 绘制纵向箱型图
# plt.xticks(rotation=45)  # 可选：旋转x轴标签以提高可读性
# plt.title("Box Plot of Each Column")  # 设置图表标题
# plt.xlabel("Columns")  # 设置x轴标签
# plt.ylabel("Values")  # 设置y轴标签
# plt.show()  # 显示图表

# 小提琴图
# sns.set(style="whitegrid")
# for column in data_x.columns:
#     plt.figure(figsize=(6, 4))  # 可选：设置每个子图的尺寸
#     sns.violinplot(data=data_x[column], orient="v")  # 绘制水平箱型图
#     plt.title(f'Box Plot of {column}')  # 设置图表标题
#     plt.xlabel(column)  # 设置x轴标签
#     plt.show()  # 显示图表

i = 0
