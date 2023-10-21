"""
__author__ = 'Cheng Yuchao'
__project__: 绘制数值的测井曲线
__time__:  2023/09/11
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
data = pd.read_csv('../data/Well3_EPOR0_1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['GR', 'NPHI', 'PEF', 'EPOR0', 'DENSITY']]
# 异常值处理：NPHI和PEF不为负值
data_x.loc[data_x['NPHI'] < 0, 'PEF'] = 0
data_x.loc[data_x['PEF'] < 0, 'PEF'] = 0
data_y = data['DEPTH']

window_size = 10  # 移动平均窗口大小
# data_x['GR'] = data_x['GR'].rolling(window=window_size).mean()
# data_x['NPHI'] = data_x['NPHI'].rolling(window=window_size).mean()
data_x['PEF'] = data_x['PEF'].rolling(window=window_size).mean()
# data_x['EPOR0'] = data_x['EPOR0'].rolling(window=window_size).mean()
# data_x['DENSITY'] = data_x['DENSITY'].rolling(window=window_size).mean()

data_lstm = pd.read_excel('../result/lstm_result.xlsx')

# 定义英尺到米的函数
def feet_to_meters(feet):
    # 定义英尺到米的转换率
    feet_to_meters_conversion = 0.3048
    # 执行转换
    meters = feet * feet_to_meters_conversion
    return meters


data_y = feet_to_meters(data_y)

# 绘图
f, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 12))
zmin = data_y.min()
zmax = data_y.max()
# 1. GR测井曲线
# ax[0].plot(data_x.GR, data_y, '-g')
# ax[0].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
# ax[0].xaxis.set_label_position('top')  # 设置标签位置为上方
# ax[0].set_xlabel("GR")
# ax[0].set_xlim(data_x.GR.min(), data_x.GR.max())  # 设置纵轴范围，范围由 logs.GR 列的最小值和最大值确定。
# 2. DENSITY测井曲线
ax[0].plot(data_x.DENSITY, data_y, '-')
ax[0].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[0].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[0].set_xlabel("DENSITY")
ax[0].set_xlim(data_x.DENSITY.min(), data_x.DENSITY.max())
# ax[0].set_yticklabels([])
# 3. NPHI测井曲线
# ax[1].plot(data_x.NPHI, data_y, '-', color='#DA70D6')
# ax[1].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
# ax[1].xaxis.set_label_position('top')  # 设置标签位置为上方
# ax[1].set_xlabel("NPHI")
# ax[1].set_xlim(data_x.NPHI.min(), data_x.NPHI.max())
# ax[1].set_yticklabels([])
# # 4.  PEF测井曲线
# ax[2].plot(data_x.PEF, data_y, '-', color='r')
# ax[2].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
# ax[2].xaxis.set_label_position('top')  # 设置标签位置为上方
# ax[2].set_xlabel("PEF")
# ax[2].set_xlim(data_x.PEF.min(), data_x.PEF.max())
# ax[2].set_yticklabels([])
# # 5. EPOR0测井曲线
# ax[3].plot(data_x.EPOR0, data_y, '-', color='blue')
# ax[3].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
# ax[3].xaxis.set_label_position('top')  # 设置标签位置为上方
# ax[3].set_xlabel("EPOR0")
# ax[3].set_xlim(data_x.EPOR0.min(), data_x.EPOR0.max())
# ax[3].set_yticklabels([])

for i in range(len(ax)):
    ax[i].set_ylim(zmin, zmax)  # 设置子图的x轴范围，即设置深度的上限和下限。
    ax[i].invert_yaxis() # 反转y轴
    # ax[i].grid(True) # 添加网格

f.suptitle('A井', fontsize=14, y=0.94)
plt.show()
