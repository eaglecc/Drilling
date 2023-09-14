"""
__author__ = 'Cheng Yuchao'
__project__:data_visualization
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
data = pd.read_csv('../data/Well4_EPOR0_1.csv')
# data.dropna(axis=0, how='any')  #只要行中包含任何一个缺失值，就删除整行。
data_x = data[['GR', 'DENSITY', 'NPHI', 'VSHALE', 'DPHI', 'EPOR0','LITH']]
data_y = data['DEPTH']

# 绘图
f, ax = plt.subplots(nrows=7, ncols=1, figsize=(12, 8))
zmin = data_y.min()
zmax = data_y.max()
# 1. GR测井曲线
ax[0].plot(data_y, data_x.GR, '-g')
ax[0].set_ylabel("GR")
ax[0].set_ylim(data_x.GR.min(), data_x.GR.max())  # 设置纵轴范围，范围由 logs.GR 列的最小值和最大值确定。
ax[0].set_xticklabels([])
# 2. DENSITY测井曲线
ax[1].plot(data_y, data_x.DENSITY, '-')
ax[1].set_ylabel("DENSITY")
ax[1].set_ylim(data_x.DENSITY.min(), data_x.DENSITY.max())
ax[1].set_xticklabels([])
# 3. NPHI测井曲线
ax[2].plot(data_y, data_x.NPHI, '-', color='#DA70D6')
ax[2].set_ylabel("NPHI")
ax[2].set_ylim(data_x.NPHI.min(), data_x.NPHI.max())
ax[2].set_xticklabels([])
# 4. VSHALE测井曲线
ax[3].plot(data_y, data_x.VSHALE, '-', color='r')
ax[3].set_ylabel("VSHALE")
ax[3].set_ylim(data_x.VSHALE.min(), data_x.VSHALE.max())
ax[3].set_xticklabels([])
# 5. DPHI测井曲线
ax[4].plot(data_y, data_x.DPHI, '-', color='#ee9922')
ax[4].set_ylabel("DPHI")
ax[4].set_ylim(data_x.DPHI.min(), data_x.DPHI.max())
ax[4].set_xticklabels([])
# 6. EPOR0测井曲线
ax[5].plot(data_y, data_x.EPOR0, '-', color='blue')
ax[5].set_ylabel("EPOR0")
ax[5].set_ylim(data_x.EPOR0.min(), data_x.EPOR0.max())
ax[5].set_xticklabels([])

# 7. 岩性色块
color_mapping = {1: '#DCDCDC', 2: '#808080', 3: '#FFD700'}# 定义岩性标签对应的颜色映射
#   初始化当前岩性标签和起始深度
current_rock_type = data_x.LITH[0]
start_depth = data_y[0]
#   遍历深度和岩性标签数据，绘制色块
for i in range(len(data_y)):
    if data_x.LITH[i] != current_rock_type:
        # 如果岩性标签发生变化，绘制色块
        end_depth = data_y[i]
        color = color_mapping[current_rock_type]
        ax[6].axvspan(start_depth,end_depth,  color=color)
        # 更新当前岩性标签和起始深度
        current_rock_type = data_x.LITH[i]
        start_depth = end_depth

#   绘制最后一个色块
end_depth = data_y.iloc[-1]
color = color_mapping[current_rock_type]
ax[6].axvspan(start_depth, end_depth, color=color)
#   设置坐标轴标签
ax[6].set_xlabel('Depth')
ax[6].set_yticklabels([])

for i in range(len(ax)):
    ax[i].set_xlim(zmin, zmax)  # 设置子图的x轴范围，即设置深度的上限和下限。

f.suptitle('Well1', fontsize=14, y=0.94)
plt.show()
