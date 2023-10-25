"""
__author__ = 'Cheng Yuchao'
__project__: 同时显示Transformer、LSTM、GRU、WLPTransformer预测结果
__time__:  2023/10/21
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
data_transformer = pd.read_excel('../result/transformer_result.xlsx')
data_WLPtransformer = pd.read_excel('../result/WLPtransformer_result.xlsx')
data_lstm = pd.read_excel('../result/lstm_result.xlsx')
data_gru = pd.read_excel('../result/gru_result.xlsx')
data = pd.read_csv('../data/Well3_EPOR0_1.csv')
data_y = data['DEPTH']
data_lith = data['LITH']


# 定义英尺到米的函数
def feet_to_meters(feet):
    # 定义英尺到米的转换率
    feet_to_meters_conversion = 0.3048
    # 执行转换
    meters = feet * feet_to_meters_conversion
    return meters


data_y = feet_to_meters(data_y)
min_value = data_y.min().min()
max_value = data_y.max().max()
train_size1 = int(0.4 * (max_value - min_value) + min_value)  # 2515m
train_size2 = int(0.6 * (max_value - min_value) + min_value)  # 2678m
# 绘图
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 12))
# zmin = data_transformer.min()
# zmax = data_transformer.max()
# 'b'：蓝色
# 'g'：绿色
# 'r'：红色
# 'c'：青色（蓝绿色）
# 'm'：洋红（紫红色）
# 'y'：黄色
# 'k'：黑色
# 'w'：白色
# 1. WLP-Transformer预测测井曲线
ax[0].plot(data_WLPtransformer.well3_DEN_predicted, data_y[80:], '-r', linewidth=1)
ax[0].plot(data_WLPtransformer.well3_DEN_true, data_y[80:], '-g', linewidth=1)
ax[0].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[0].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[0].set_xlabel("WLP-Transformer")

# 2. Transformer预测测井曲线
ax[1].plot(data_transformer.well3_DEN_predicted, data_y[80:], '-r', linewidth=1)
ax[1].plot(data_transformer.well3_DEN_true, data_y[80:], '-g', linewidth=1)
ax[1].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[1].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[1].set_xlabel("Transformer")
ax[1].set_yticklabels([])

# 3. lstm测井曲线
ax[2].plot(data_lstm.lstm_DENSITY_predicted, data_y[80:], '-r', linewidth=1)
ax[2].plot(data_lstm.lstm_DENSITY_true, data_y[80:], '-g', linewidth=1)
ax[2].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[2].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[2].set_xlabel("LSTM")
ax[2].set_yticklabels([])
# 4. GRU测井曲线
ax[3].plot(data_gru.gru_DENSITY_predicted, data_y[80:], '-r', linewidth=1)
ax[3].plot(data_gru.gru_DENSITY_true, data_y[80:], '-g', linewidth=1)
ax[3].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
ax[3].xaxis.set_label_position('top')  # 设置标签位置为上方
ax[3].set_xlabel("GRU")
ax[3].set_yticklabels([])
# # 4.  PEF测井曲线
# ax[2].plot(data_x.PEF, data_y, '-', color='r')
# ax[2].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
# ax[2].xaxis.set_label_position('top')  # 设置标签位置为上方
# ax[2].set_xlabel("PEF")
# ax[2].set_xlim(data_x.PEF.min(), data_x.PEF.max())
# ax[2].set_yticklabels([])
# # 5. EPOR0测井曲线
# ax[3].plot(data_x.EPOR0, data_y, '-', color='blue')
# ax[3].xaxis.set_ticks_position('top')  # 设置刻度位置为上方
# ax[3].xaxis.set_label_position('top')  # 设置标签位置为上方
# ax[3].set_xlabel("EPOR0")
# ax[3].set_xlim(data_x.EPOR0.min(), data_x.EPOR0.max())
# ax[3].set_yticklabels([])

for i in range(len(ax)):
    # ax[i].set_ylim(zmin, zmax)  # 设置子图的x轴范围，即设置深度的上限和下限。
    ax[i].invert_yaxis()  # 反转y轴
    # ax[i].grid(True) # 添加网格
    # ax[i].set_ylim(train_size1, train_size2)  # 局部放大

f.suptitle('C井密度预测', fontsize=14, y=0.94)
plt.show()

# 计算评价指标
# 1. WLPtransformer
lower_bound = 0.4  # 40%
upper_bound = 0.6  # 60%
start_index = int(lower_bound * len(data_WLPtransformer.well3_DEN_true))
end_index = int(upper_bound * len(data_WLPtransformer.well3_DEN_true))
mse = np.mean((data_WLPtransformer.well3_DEN_true[start_index:end_index] - data_WLPtransformer.well3_DEN_predicted[start_index:end_index]) ** 2)
rmse = np.sqrt(np.mean((data_WLPtransformer.well3_DEN_true[start_index:end_index] - data_WLPtransformer.well3_DEN_predicted[start_index:end_index]) ** 2))
mae = np.mean(np.abs(data_WLPtransformer.well3_DEN_true[start_index:end_index] - data_WLPtransformer.well3_DEN_predicted[start_index:end_index]))
mape = np.mean(np.abs((data_WLPtransformer.well3_DEN_true[start_index:end_index] - data_WLPtransformer.well3_DEN_predicted[start_index:end_index]) / data_WLPtransformer.well3_DEN_true[start_index:end_index]))
print("MSE", mse)
print("RMSE", rmse)
print("MAPE:", mae)
print("MAPE:", mape)
# 创建一个txt文件并将结果写入其中
resultpath = '../result/评价指标WLPtransformer.txt'
with open(resultpath, "w") as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"MAPE: {mape}\n")

# 2. transformer
mse = np.mean((data_transformer.well3_DEN_true[start_index:end_index] - data_transformer.well3_DEN_predicted[start_index:end_index]) ** 2)
rmse = np.sqrt(np.mean((data_transformer.well3_DEN_true[start_index:end_index] - data_transformer.well3_DEN_predicted[start_index:end_index]) ** 2))
mae = np.mean(np.abs(data_transformer.well3_DEN_true[start_index:end_index] - data_transformer.well3_DEN_predicted[start_index:end_index]))
mape = np.mean(
    np.abs((data_transformer.well3_DEN_true[start_index:end_index] - data_transformer.well3_DEN_predicted[start_index:end_index]) / data_transformer.well3_DEN_true[start_index:end_index]))
print("MSE", mse)
print("RMSE", rmse)
print("MAPE:", mae)
print("MAPE:", mape)
# 创建一个txt文件并将结果写入其中
resultpath = ('../result/评价指标transformer.txt')
with open(resultpath, "w") as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"MAPE: {mape}\n")

# 3. lstm
mse = np.mean((data_lstm.lstm_DENSITY_true[start_index:end_index] - data_lstm.lstm_DENSITY_predicted[start_index:end_index]) ** 2)
rmse = np.sqrt(np.mean((data_lstm.lstm_DENSITY_true[start_index:end_index] - data_lstm.lstm_DENSITY_predicted[start_index:end_index]) ** 2))
mae = np.mean(np.abs(data_lstm.lstm_DENSITY_true[start_index:end_index] - data_lstm.lstm_DENSITY_predicted[start_index:end_index]))
mape = np.mean(np.abs((data_lstm.lstm_DENSITY_true[start_index:end_index] - data_lstm.lstm_DENSITY_predicted[start_index:end_index]) / data_lstm.lstm_DENSITY_true[start_index:end_index]))
print("MSE", mse)
print("RMSE", rmse)
print("MAPE:", mae)
print("MAPE:", mape)
# 创建一个txt文件并将结果写入其中
resultpath = ('../result/评价指标lstm.txt')
with open(resultpath, "w") as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"MAPE: {mape}\n")

# 4. gru
mse = np.mean((data_gru.gru_DENSITY_true[start_index:end_index] - data_gru.gru_DENSITY_predicted[start_index:end_index]) ** 2)
rmse = np.sqrt(np.mean((data_gru.gru_DENSITY_true[start_index:end_index] - data_gru.gru_DENSITY_predicted[start_index:end_index]) ** 2))
mae = np.mean(np.abs(data_gru.gru_DENSITY_true[start_index:end_index] - data_gru.gru_DENSITY_predicted[start_index:end_index]))
mape = np.mean(
    np.abs((data_gru.gru_DENSITY_true[start_index:end_index] - data_gru.gru_DENSITY_predicted[start_index:end_index]) / data_gru.gru_DENSITY_true[start_index:end_index]))
print("MSE", mse)
print("RMSE", rmse)
print("MAPE:", mae)
print("MAPE:", mape)
# 创建一个txt文件并将结果写入其中
resultpath = ('../result/评价指标gru.txt')
with open(resultpath, "w") as file:
    file.write(f"MSE: {mse}\n")
    file.write(f"RMSE: {rmse}\n")
    file.write(f"MAE: {mae}\n")
    file.write(f"MAPE: {mape}\n")
