"""
__author__ = 'Cheng Yuchao'
__project__: 大庆油田 可视化
__time__:  2023/10/25
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
data = pd.read_csv('../data/daqingyoutian/vertical_all_A3.csv')
data_x = data[['RMN-RMG',  'HAC     .us/m', 'BHC     .','CAL     .cm', 'SP      .mv', 'GR      .','DEN     .g/cm3']].values
data_y = data['DEPT    .M']


zmin = data_y.min()
zmax = data_y.max()
min_value = data_y.min().min()
max_value = data_y.max().max()
train_size1 = int(0.95 * (max_value - min_value) + min_value)
train_size2 = int(1 * (max_value - min_value) + min_value)

f, ax = plt.subplots(nrows=1, ncols=7, figsize=(8, 12))
# 1. RMN-RMG
ax[0].plot(data_x[:, 0], data_y, '-b', linewidth=0.8)
ax[0].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[0].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[0].set_xlabel("RMN-RMG")

# 2. HAC
ax[1].plot(data_x[:, 1], data_y, '-b', linewidth=0.8)
ax[1].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[1].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[1].set_xlabel("HAC")
ax[1].set_yticklabels([])

# 3. BHC
ax[2].plot(data_x[:, 2], data_y, '-b', linewidth=0.8)
ax[2].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[2].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[2].set_xlabel("BHC")
ax[2].set_yticklabels([])

# 4. CAL
ax[3].plot(data_x[:, 3], data_y, '-b', linewidth=0.8)
ax[3].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[3].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[3].set_xlabel("CAL")
ax[3].set_yticklabels([])

# 5. SP
ax[4].plot(data_x[:, 4], data_y, '-b', linewidth=0.8)
ax[4].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[4].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[4].set_xlabel("SP")
ax[4].set_yticklabels([])
#
# 6. GR
ax[5].plot(data_x[:, 5], data_y, '-b', linewidth=0.8)
ax[5].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[5].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[5].set_xlabel("GR")
ax[5].set_yticklabels([])
#
# 7. DEN
ax[6].plot(data_x[:, 6], data_y, '-b', linewidth=0.8)
ax[6].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[6].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[6].set_xlabel("DEN")
ax[6].set_yticklabels([])

for i in range(len(ax)):
    ax[i].set_ylim(zmin, zmax)  # 设置子图的x轴范围，即设置深度的上限和下限。
    # ax[i].set_ylim(train_size1, train_size2)  # 局部放大
    # ax[i].set_ylim(1000, train_size2)  # 局部放大
    ax[i].invert_yaxis()  # 反转y轴
    # ax[i].grid(True) # 添加网格

f.suptitle('大庆油田C井测井曲线', fontsize=14, y=0.94)
plt.show()
