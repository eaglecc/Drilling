"""
__author__ = 'Cheng Yuchao'
__project__: 测井曲线相关性分析
__time__:  2023/10/12
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
data = pd.read_csv('../data/Well3_EPOR0_1.csv')
data_x = data[['DEPTH', 'GR', 'NPHI', 'DENSITY', 'PEF', 'LITH', 'VSHALE', 'DPHI', 'EPOR0']]
data_y = data['DEPTH']

correlation_matrix = data_x.corr()

plt.figure(figsize=(10, 8))
# ＃cmap有很多可选的“色号”，如：YlGnBu，YlGnBu＿r，hot，hot＿r，OrRd，autumn，greens，viridis，greys，Purples，rainbow，gist＿rainbow。
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap="BuPu", linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", linewidths=.5)
sns.heatmap(correlation_matrix, annot=True, cmap="Purples", linewidths=.5)


plt.title('Pearson Correlation Heatmap')
plt.show()


