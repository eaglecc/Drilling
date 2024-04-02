"""
__author__ = 'Cheng Yuchao'
__project__: 大庆油田 可视化  A井数据上对比 LSTM、GRU、TRansformer
# 计算  MAE、RMSE、R2
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
data_lstm_DEN = pd.read_excel('../result/wlp_transformer/daqingyoutian_A_lstm_DEN_result.xlsx')
data_lstm_BHC = pd.read_excel('../result/wlp_transformer/daqingyoutian_A_lstm_BHC_result.xlsx')
data_gru_DEN = pd.read_excel('../result/wlp_transformer/daqingyoutian_GRU_DEN_result.xlsx')
data_gru_BHC = pd.read_excel('../result/wlp_transformer/daqingyoutian_GRU_BHC_result.xlsx')
data_wlptransformer_DEN = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_DEN.xlsx')
data_wlptransformer_BHC = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_BHC.xlsx')
data_transformer_DEN = pd.read_excel('../result/wlp_transformer/transformer_result_DEN.xlsx')
data_transformer_BHC = pd.read_excel('../result/wlp_transformer/transformer_result_BHC.xlsx')


# 计算 DEN: MAE、RMSE、R2
lstm_rmse = np.sqrt(np.mean((data_lstm_DEN.lstm_DEN_predicted - data_lstm_DEN.lstm_DEN_true) ** 2))
lstm_mae = np.mean(np.abs(data_lstm_DEN.lstm_DEN_predicted - data_lstm_DEN.lstm_DEN_true))
mean_y_true = np.mean(data_lstm_DEN.lstm_DEN_true)
ss_total = np.sum((data_lstm_DEN.lstm_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_lstm_DEN.lstm_DEN_true - data_lstm_DEN.lstm_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("LSTM --- DEN")
print(lstm_rmse)
print(lstm_mae)
print(r2)

gru_rmse = np.sqrt(np.mean((data_gru_DEN.gru_DEN_predicted - data_gru_DEN.gru_DEN_true) ** 2))
gru_mae = np.mean(np.abs(data_gru_DEN.gru_DEN_predicted - data_gru_DEN.gru_DEN_true))
mean_y_true = np.mean(data_gru_DEN.gru_DEN_true)
ss_total = np.sum((data_gru_DEN.gru_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_gru_DEN.gru_DEN_true -data_gru_DEN.gru_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("GRU --- DEN")
print(gru_rmse)
print(gru_mae)
print(r2)

transformer_rmse = np.sqrt(np.mean((data_transformer_DEN.well1_DEN_predicted - data_transformer_DEN.well1_DEN_true) ** 2))
transformer_mae = np.mean(np.abs(data_transformer_DEN.well1_DEN_predicted - data_transformer_DEN.well1_DEN_true))
mean_y_true = np.mean(data_transformer_DEN.well1_DEN_true)
ss_total = np.sum((data_transformer_DEN.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_transformer_DEN.well1_DEN_true - data_transformer_DEN.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("T --- DEN")
print(transformer_rmse)
print(transformer_mae)
print(r2)

wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN.well1_DEN_predicted - data_wlptransformer_DEN.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN.well1_DEN_predicted - data_wlptransformer_DEN.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN.well1_DEN_true -data_wlptransformer_DEN.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)

# 计算 BHC: MAE、RMSE、R2
lstm_rmse = np.sqrt(np.mean((data_lstm_BHC.lstm_BHC_predicted - data_lstm_BHC.lstm_BHC_true) ** 2))
lstm_mae = np.mean(np.abs(data_lstm_BHC.lstm_BHC_predicted - data_lstm_BHC.lstm_BHC_true))
mean_y_true = np.mean(data_lstm_BHC.lstm_BHC_true)
ss_total = np.sum((data_lstm_BHC.lstm_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_lstm_BHC.lstm_BHC_true - data_lstm_BHC.lstm_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("LSTM --- BHC")
print(lstm_rmse)
print(lstm_mae)
print(r2)

gru_rmse = np.sqrt(np.mean((data_gru_BHC.gru_BHC_predicted - data_gru_BHC.gru_BHC_true) ** 2))
gru_mae = np.mean(np.abs(data_gru_BHC.gru_BHC_predicted - data_gru_BHC.gru_BHC_true))
mean_y_true = np.mean(data_gru_BHC.gru_BHC_true)
ss_total = np.sum((data_gru_BHC.gru_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_gru_BHC.gru_BHC_true - data_gru_BHC.gru_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("GRU --- BHC")
print(gru_rmse)
print(gru_mae)
print(r2)

transformer_rmse = np.sqrt(np.mean((data_transformer_BHC.well1_BHC_predicted - data_transformer_BHC.well1_BHC_true) ** 2))
transformer_mae = np.mean(np.abs(data_transformer_BHC.well1_BHC_predicted - data_transformer_BHC.well1_BHC_true))
mean_y_true = np.mean(data_transformer_BHC.well1_BHC_true)
ss_total = np.sum((data_transformer_BHC.well1_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_transformer_BHC.well1_BHC_true - data_transformer_BHC.well1_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("T --- BHC")
print(transformer_rmse)
print(transformer_mae)
print(r2)

wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_BHC.well1_BHC_predicted - data_wlptransformer_BHC.well1_BHC_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_BHC.well1_BHC_predicted - data_wlptransformer_BHC.well1_BHC_true))
mean_y_true = np.mean(data_wlptransformer_BHC.well1_BHC_true)
ss_total = np.sum((data_wlptransformer_BHC.well1_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_BHC.well1_BHC_true - data_wlptransformer_BHC.well1_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- BHC")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)


