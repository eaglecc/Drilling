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
data = pd.read_csv('../data/daqingyoutian/vertical_all_A1.csv')
data_x = data[['RMN-RMG',  'HAC     .us/m', 'BHC     .','CAL     .cm ', 'SP      .mv  ', 'GR      .   ',
               'DEN     .g/cm3 ']].values
data_y = data['DEPT    .M ']
data_DEN = pd.read_excel('../result/daqingyoutian_result_DEN.xlsx')
data_CAL = pd.read_excel('../result/daqingyoutian_result_CAL.xlsx')
data_GR = pd.read_excel('../result/daqingyoutian_result_GR.xlsx')
data_SP = pd.read_excel('../result/daqingyoutian_result_SP.xlsx')
data_lstm_DEN = pd.read_excel('../result/daqingyoutian_lstm_DEN_result.xlsx')
data_lstm_CAL = pd.read_excel('../result/daqingyoutian_lstm_CAL_result.xlsx')
data_lstm_GR = pd.read_excel('../result/daqingyoutian_lstm_GR_result.xlsx')
data_lstm_SP = pd.read_excel('../result/daqingyoutian_lstm_SP_result.xlsx')
data_gru_DEN = pd.read_excel('../result/daqingyoutian_GRU_DEN_result.xlsx')
data_gru_CAL = pd.read_excel('../result/daqingyoutian_GRU_CAL_result.xlsx')
data_gru_GR  = pd.read_excel('../result/daqingyoutian_GRU_GR_result.xlsx')
data_gru_SP  = pd.read_excel('../result/daqingyoutian_GRU_SP_result.xlsx')

zmin = data_y.min()
zmax = data_y.max()
min_value = data_y.min().min()
max_value = data_y.max().max()
train_size1 = int(0.95 * (max_value - min_value) + min_value)
train_size2 = int(1 * (max_value - min_value) + min_value)

f, ax = plt.subplots(nrows=1, ncols=7, figsize=(8, 12))
# 1. RMN-RMG
ax[0].plot(data_x[:, 0], data_y, '-g', linewidth=0.8)
ax[0].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[0].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[0].set_xlabel("RMN-RMG")

# 2. HAC
ax[1].plot(data_x[:, 1], data_y, '-g', linewidth=0.8)
ax[1].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[1].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[1].set_xlabel("HAC")
ax[1].set_yticklabels([])

# 3. BHC
ax[2].plot(data_x[:, 2], data_y, '-g', linewidth=0.8)
ax[2].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[2].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[2].set_xlabel("BHC")
ax[2].set_yticklabels([])

# 4. CAL
ax[3].plot(data_x[:, 3], data_y, '-g', linewidth=0.8)
ax[3].plot(data_CAL.well1_CAL_predicted, data_y[-len(data_CAL.well1_CAL_predicted):], '-r', linewidth=0.8)
ax[3].plot(data_lstm_CAL.lstm_CAL_predicted, data_y[-len(data_lstm_CAL.lstm_CAL_predicted):], '-b', linewidth=0.8)
ax[3].plot(data_gru_CAL.gru_CAL_predicted, data_y[-len(data_gru_CAL.gru_CAL_predicted):], '-k', linewidth=0.8)
ax[3].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[3].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[3].set_xlabel("CAL")
ax[3].set_yticklabels([])

# 5. SP
ax[4].plot(data_x[:, 4], data_y, '-g', linewidth=0.8)
ax[4].plot(data_SP.well1_SP_predicted, data_y[-len(data_SP.well1_SP_predicted):], '-r', linewidth=0.8)
ax[4].plot(data_lstm_SP.lstm_SP_predicted, data_y[-len(data_lstm_SP.lstm_SP_predicted):], '-b', linewidth=0.8)
ax[4].plot(data_gru_SP.gru_SP_predicted, data_y[-len(data_gru_SP.gru_SP_predicted):], '-k', linewidth=0.8)
ax[4].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[4].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[4].set_xlabel("SP")
ax[4].set_yticklabels([])

# 6. GR
ax[5].plot(data_x[:, 5], data_y, '-g', linewidth=0.8)
ax[5].plot(data_GR.well1_GR_predicted, data_y[-len(data_GR.well1_GR_predicted):], '-r', linewidth=0.8)
ax[5].plot(data_lstm_GR.lstm_GR_predicted, data_y[-len(data_lstm_GR.lstm_GR_predicted):], '-b', linewidth=0.8)
ax[5].plot(data_gru_GR.gru_GR_predicted, data_y[-len(data_gru_GR.gru_GR_predicted):], '-k', linewidth=0.8)
ax[5].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[5].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[5].set_xlabel("GR")
ax[5].set_yticklabels([])

# 7. DEN
ax[6].plot(data_x[:, 6], data_y, '-g', linewidth=0.8)
ax[6].plot(data_DEN.well1_DEN_predicted, data_y[-len(data_DEN.well1_DEN_predicted):], '-r', linewidth=0.8)
ax[6].plot(data_lstm_DEN.lstm_DEN_predicted, data_y[-len(data_lstm_DEN.lstm_DEN_predicted):], '-b', linewidth=0.8)
ax[6].plot(data_gru_DEN.gru_DEN_predicted, data_y[-len(data_gru_DEN.gru_DEN_predicted):], '-k', linewidth=0.8)
ax[6].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[6].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[6].set_xlabel("DEN")
ax[6].set_yticklabels([])

for i in range(len(ax)):
    # ax[i].set_ylim(zmin, zmax)  # 设置子图的x轴范围，即设置深度的上限和下限。
    ax[i].set_ylim(train_size1, train_size2)  # 局部放大
    ax[i].invert_yaxis()  # 反转y轴
    # ax[i].grid(True) # 添加网格

f.suptitle('C井密度预测', fontsize=14, y=0.94)
plt.show()
