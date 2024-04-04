"""
__author__ = 'Cheng Yuchao'
__project__: 大庆油田 缺失数据 评价指标对比
# 计算  MAE、RMSE、R2
__time__:  2024/4/2
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

data_wlptransformer_DEN_A = pd.read_excel('../result/wlp_transformer/daqingyoutian_result缺失_DEN.xlsx')
data_wlptransformer_BHC_A = pd.read_excel('../result/wlp_transformer/daqingyoutian_result缺失_BHC.xlsx')
data_LSTM_DEN_A = pd.read_excel('../result/wlp_transformer/缺失预测A_lstm_DEN_result.xlsx')
data_LSTM_BHC_A = pd.read_excel('../result/wlp_transformer/缺失预测A_lstm_BHC_result.xlsx')
data_GRU_DEN_A = pd.read_excel('../result/wlp_transformer/缺失预测A_gru_DEN_result.xlsx')
data_GRU_BHC_A = pd.read_excel('../result/wlp_transformer/缺失预测A_gru_BHC_result.xlsx')
data_transformer_DEN_A = pd.read_excel('../result/wlp_transformer/daqingyoutian_result缺失_DEN_原始T.xlsx')
data_transformer_BHC_A = pd.read_excel('../result/wlp_transformer/daqingyoutian_result缺失_BHC_原始T.xlsx')



# 计算 DEN: MAE、RMSE、R2
wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN_A.well1_DEN_predicted - data_wlptransformer_DEN_A.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN_A.well1_DEN_predicted - data_wlptransformer_DEN_A.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN_A.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN_A.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN_A.well1_DEN_true -data_wlptransformer_DEN_A.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN---A")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)

wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_BHC_A.well1_BHC_predicted - data_wlptransformer_BHC_A.well1_BHC_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_BHC_A.well1_BHC_predicted - data_wlptransformer_BHC_A.well1_BHC_true))
mean_y_true = np.mean(data_wlptransformer_BHC_A.well1_BHC_true)
ss_total = np.sum((data_wlptransformer_BHC_A.well1_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_BHC_A.well1_BHC_true -data_wlptransformer_BHC_A.well1_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- BHC---A")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)

transformer_rmse = np.sqrt(np.mean((data_transformer_DEN_A.well1_DEN_predicted - data_transformer_DEN_A.well1_DEN_true) ** 2))
transformer_mae = np.mean(np.abs(data_transformer_DEN_A.well1_DEN_predicted - data_transformer_DEN_A.well1_DEN_true))
mean_y_true = np.mean(data_transformer_DEN_A.well1_DEN_true)
ss_total = np.sum((data_transformer_DEN_A.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_transformer_DEN_A.well1_DEN_true -data_transformer_DEN_A.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("T --- DEN---A")
print(transformer_rmse)
print(transformer_mae)
print(r2)

transformer_rmse = np.sqrt(np.mean((data_transformer_BHC_A.well1_BHC_predicted - data_transformer_BHC_A.well1_BHC_true) ** 2))
transformer_mae = np.mean(np.abs(data_transformer_BHC_A.well1_BHC_predicted - data_transformer_BHC_A.well1_BHC_true))
mean_y_true = np.mean(data_transformer_BHC_A.well1_BHC_true)
ss_total = np.sum((data_transformer_BHC_A.well1_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_transformer_BHC_A.well1_BHC_true -data_transformer_BHC_A.well1_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("T --- BHC---A")
print(transformer_rmse)
print(transformer_mae)
print(r2)

lstm_den_rmse = np.sqrt(np.mean((data_LSTM_DEN_A.lstm_DEN_predicted - data_LSTM_DEN_A.lstm_DEN_true) ** 2))
lstm_den_mae = np.mean(np.abs(data_LSTM_DEN_A.lstm_DEN_predicted - data_LSTM_DEN_A.lstm_DEN_true))
mean_y_true = np.mean(data_LSTM_DEN_A.lstm_DEN_true)
ss_total = np.sum((data_LSTM_DEN_A.lstm_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_LSTM_DEN_A.lstm_DEN_true -data_LSTM_DEN_A.lstm_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("LSTM --- DEN---A")
print(lstm_den_rmse)
print(lstm_den_mae)
print(r2)


lstm_bhc_rmse = np.sqrt(np.mean((data_LSTM_BHC_A.lstm_BHC_predicted - data_LSTM_BHC_A.lstm_BHC_true) ** 2))
lstm_bhc_mae = np.mean(np.abs(data_LSTM_BHC_A.lstm_BHC_predicted - data_LSTM_BHC_A.lstm_BHC_true))
mean_y_true = np.mean(data_LSTM_BHC_A.lstm_BHC_true)
ss_total = np.sum((data_LSTM_BHC_A.lstm_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_LSTM_BHC_A.lstm_BHC_true -data_LSTM_BHC_A.lstm_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("LSTM --- BHC ---A")
print(lstm_bhc_rmse)
print(lstm_bhc_mae)
print(r2)


gru_den_rmse = np.sqrt(np.mean((data_GRU_DEN_A.gru_DEN_predicted - data_GRU_DEN_A.gru_DEN_true) ** 2))
gru_den_mae = np.mean(np.abs(data_GRU_DEN_A.gru_DEN_predicted - data_GRU_DEN_A.gru_DEN_true))
mean_y_true = np.mean(data_GRU_DEN_A.gru_DEN_true)
ss_total = np.sum((data_GRU_DEN_A.gru_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_GRU_DEN_A.gru_DEN_true -data_GRU_DEN_A.gru_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("GRU --- DEN---A")
print(gru_den_rmse)
print(gru_den_mae)
print(r2)


gru_bhc_rmse = np.sqrt(np.mean((data_GRU_BHC_A.gru_BHC_predicted - data_GRU_BHC_A.gru_BHC_true) ** 2))
gru_bhc_mae = np.mean(np.abs(data_GRU_BHC_A.gru_BHC_predicted - data_GRU_BHC_A.gru_BHC_true))
mean_y_true = np.mean(data_GRU_BHC_A.gru_BHC_true)
ss_total = np.sum((data_GRU_BHC_A.gru_BHC_true - mean_y_true)**2)
ss_residual = np.sum((data_GRU_BHC_A.gru_BHC_true -data_GRU_BHC_A.gru_BHC_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("GRU --- BHC ---A")
print(gru_bhc_rmse)
print(gru_bhc_mae)
print(r2)