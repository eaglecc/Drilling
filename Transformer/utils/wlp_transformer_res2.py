"""
__author__ = 'Cheng Yuchao'
__project__: 大庆油田 可视化  B C D E F井数据上对比
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

data_wlptransformer_DEN_B = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_B_DEN.xlsx')
data_wlptransformer_DEN_C = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_C_DEN.xlsx')
data_wlptransformer_DEN_D = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_D_DEN.xlsx')
data_wlptransformer_DEN_E = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_E_DEN.xlsx')
data_wlptransformer_DEN_F = pd.read_excel('../result/wlp_transformer/daqingyoutian_result_F_DEN.xlsx')

# 计算 DEN: MAE、RMSE、R2
wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN_B.well1_DEN_predicted - data_wlptransformer_DEN_B.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN_B.well1_DEN_predicted - data_wlptransformer_DEN_B.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN_B.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN_B.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN_B.well1_DEN_true -data_wlptransformer_DEN_B.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN---B")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)

wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN_C.well1_DEN_predicted - data_wlptransformer_DEN_C.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN_C.well1_DEN_predicted - data_wlptransformer_DEN_C.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN_C.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN_C.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN_C.well1_DEN_true -data_wlptransformer_DEN_C.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN---C")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)


wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN_D.well1_DEN_predicted - data_wlptransformer_DEN_D.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN_D.well1_DEN_predicted - data_wlptransformer_DEN_D.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN_D.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN_D.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN_D.well1_DEN_true -data_wlptransformer_DEN_D.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN---D")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)


wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN_E.well1_DEN_predicted - data_wlptransformer_DEN_E.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN_E.well1_DEN_predicted - data_wlptransformer_DEN_E.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN_E.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN_E.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN_E.well1_DEN_true -data_wlptransformer_DEN_E.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN---E")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)

wlptransformer_rmse = np.sqrt(np.mean((data_wlptransformer_DEN_F.well1_DEN_predicted - data_wlptransformer_DEN_F.well1_DEN_true) ** 2))
wlptransformer_mae = np.mean(np.abs(data_wlptransformer_DEN_F.well1_DEN_predicted - data_wlptransformer_DEN_F.well1_DEN_true))
mean_y_true = np.mean(data_wlptransformer_DEN_F.well1_DEN_true)
ss_total = np.sum((data_wlptransformer_DEN_F.well1_DEN_true - mean_y_true)**2)
ss_residual = np.sum((data_wlptransformer_DEN_F.well1_DEN_true -data_wlptransformer_DEN_F.well1_DEN_predicted)**2)
r2 = 1 - (ss_residual / ss_total)
print("WLP-T --- DEN---F")
print(wlptransformer_rmse)
print(wlptransformer_mae)
print(r2)
