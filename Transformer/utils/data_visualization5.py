"""
__author__ = 'Cheng Yuchao'
__project__: 大庆油田 可视化  ：单条测井数据
__time__:  2023/12/13
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
data_x = data[['RMN-RMG', 'HAC     .us/m', 'BHC     .', 'CAL     .cm ', 'SP      .mv  ', 'GR      .   ',
               'DEN     .g/cm3 ']].values
data_y = data['DEPT    .M ']
data_DEN = pd.read_excel('../result/daqingyoutian_result_DEN.xlsx').head(320)
data_CAL = pd.read_excel('../result/daqingyoutian_result_CAL.xlsx').head(320)
data_GR = pd.read_excel('../result/daqingyoutian_result_GR.xlsx').head(320)
data_SP = pd.read_excel('../result/daqingyoutian_result_SP.xlsx').head(320)
data_lstm_DEN = pd.read_excel('../result/daqingyoutian_lstm_DEN_result.xlsx').head(320)
data_lstm_CAL = pd.read_excel('../result/daqingyoutian_lstm_CAL_result.xlsx').head(320)
data_lstm_GR = pd.read_excel('../result/daqingyoutian_lstm_GR_result.xlsx').head(320)
data_lstm_SP = pd.read_excel('../result/daqingyoutian_lstm_SP_result.xlsx').head(320)
data_gru_DEN = pd.read_excel('../result/daqingyoutian_GRU_DEN_result.xlsx').head(320)
data_gru_CAL = pd.read_excel('../result/daqingyoutian_GRU_CAL_result.xlsx').head(320)
data_gru_GR = pd.read_excel('../result/daqingyoutian_GRU_GR_result.xlsx').head(320)
data_gru_SP = pd.read_excel('../result/daqingyoutian_GRU_SP_result.xlsx').head(320)
data_transformer_GR = pd.read_excel('../result/transformer_daqingyoutian_result_GR.xlsx').head(320)
data_transformer_SP = pd.read_excel('../result/transformer_daqingyoutian_result_SP.xlsx').head(320)
data_transformer_CAL = pd.read_excel('../result/transformer_daqingyoutian_result_CAL.xlsx').head(320)
data_transformer_DEN = pd.read_excel('../result/transformer_daqingyoutian_result_DEN.xlsx').head(320)

col0_den = data_DEN['well1_DEN_true']
col1_den = data_DEN['well1_DEN_predicted']
col2_den = data_lstm_DEN['lstm_DEN_predicted']
col3_den = data_gru_DEN['gru_DEN_predicted']
col4_den = data_transformer_DEN['well1_DEN_predicted']
df_den = pd.concat([col0_den, col1_den, col2_den, col3_den, col4_den], axis=1)
new_columns = ['True', 'WLP_Transformer', 'LSTM', 'GRU', 'Transformer']
df_den.columns = new_columns

# 创建三个子图
fig, axes = plt.subplots(4, 1, figsize=(12, 18))

df_den.plot(y=['True', 'WLP_Transformer'], ax=axes[0], style=['#CFCFCF', 'r--'])
# axes[0].set_title('真实值和WLP_Transformer预测结果对比')
df_den.plot(y=['True', 'LSTM'], ax=axes[1], style=['#CFCFCF', 'g-.'])
# axes[1].set_title('真实值和LSTM预测结果对比')

df_den.plot(y=['True', 'GRU'], ax=axes[2], style=['#CFCFCF', 'b--'])
# axes[2].set_title('真实值和GRU预测结果对比')

df_den.plot(y=['True', 'Transformer'], ax=axes[3], style=['#CFCFCF', 'm-.'])
# axes[3].set_title('真实值和Transformer预测结果对比')

plt.tight_layout()
plt.show()
