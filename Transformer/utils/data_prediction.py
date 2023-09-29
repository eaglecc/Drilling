"""
__author__ = 'Cheng Yuchao'
__project__: 数据预处理
__time__:  2023/09/27
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
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data[['GR', 'DENSITY', 'NPHI', 'VSHALE', 'DPHI', 'EPOR0','LITH']]
data_y = data['DEPTH']