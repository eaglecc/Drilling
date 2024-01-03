"""
__author__ = 'Cheng Yuchao'
__project__: 测井曲线相关性分析:大庆油田
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
data_x = data[
    ['RMN-RMG', 'CAL     .cm ', 'SP      .mv  ', 'GR      .   ', 'DEN     .g/cm3 ', 'HAC     .us/m', 'BHC     .']]
data_y = data['DEPT    .M ']

correlation_matrix = data_x.corr()
# 属性的自定义标题
attribute_names = ["RMN-RMG", "CAL", "SP", "GR", "DEN", "HAC", "BHC"]

plt.figure(figsize=(10, 8))
# ＃cmap有很多可选的“色号”，如：YlGnBu，YlGnBu＿r，hot，hot＿r，OrRd，autumn，greens，viridis，greys，Purples，rainbow，gist＿rainbow。
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap='viridis', linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap="BuPu", linewidths=.5)
# sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", linewidths=.5)
sns.heatmap(correlation_matrix, annot=True, cmap="Purples", linewidths=.5
            , xticklabels=attribute_names, yticklabels=attribute_names)

plt.title('Pearson Correlation Heatmap')
plt.show()
