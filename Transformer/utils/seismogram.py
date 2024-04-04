"""
__author__ = 'Cheng Yuchao'
__project__: 合成地震曲线
__time__:  2024/4/2
__email__:"2477721334@qq.com"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import warnings
from scipy.signal import convolve, ricker

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入数据
data = pd.read_csv('../data/daqingyoutian/vertical_all_A1.csv')
data_x = data[['RMN-RMG',  'HAC     .us/m', 'BHC     .','CAL     .cm ', 'SP      .mv  ', 'GR      .   ',
               'DEN     .g/cm3 ']]
data_y = data['DEPT    .M ']
train_size1 = int(0.4 * (len(data_x) - 500))
train_size2 = int(0.6 * (len(data_x) - 500))
depth = data_y[train_size1:train_size2].reset_index(drop=True)
data_BHC = pd.read_excel('../result/wlp_transformer/daqingyoutian_result校正_BHC.xlsx')
data_DEN = pd.read_excel('../result/wlp_transformer/daqingyoutian_result校正_DEN.xlsx')


# sonic_log = data_x['BHC     .']  # 全部声波速度数据
# density_log = data_x['DEN     .g/cm3 ']  # 全部密度数据
# sonic_log = data_BHC['well1_BHC_predicted']  # 声波速度数据
# density_log = data_DEN['well1_DEN_predicted']  # 密度数据
sonic_log = data_BHC['well1_BHC_true']  # 全部声波速度数据
density_log =  data_DEN['well1_DEN_true'] # 全部密度数据
velocity_log = 1e6 / sonic_log  # 转换为m/s

# 计算声阻抗
acoustic_impedance = velocity_log * density_log

# 计算反射系数
reflection_coefficients = (np.roll(acoustic_impedance, -1) - acoustic_impedance)[:-1] / \
                          (np.roll(acoustic_impedance, -1) + acoustic_impedance)[:-1]

# 生成雷克子波
points = 100  # 子波点数，影响子波的长度
a = 4  # 雷克子波参数，控制子波的宽度
wavelet = ricker(points, a)

# 执行卷积以生成合成地震记录
synthetic_seismogram = convolve(reflection_coefficients, wavelet, mode='same')
synthetic_seismogram = synthetic_seismogram[:len(reflection_coefficients)]  # 裁剪以匹配反射系数数组长度

# 规范化合成地震记录以便于显示
synthetic_seismogram /= np.max(np.abs(synthetic_seismogram))

# 存入excel
file_name = '../result/wlp_transformer/真实振幅.xlsx'
df = pd.DataFrame()  # 创建一个新 DataFrame
df['depth'] = data_DEN['depth'][:-1]
df['Amplitude'] = synthetic_seismogram
df.to_excel(file_name, index=False)  # index=False 防止写入索引

# 绘制地震记录
plt.figure(figsize=(4, 10))
plt.plot(synthetic_seismogram, np.arange(len(synthetic_seismogram)), color='black')
plt.fill_betweenx(np.arange(len(synthetic_seismogram)), synthetic_seismogram, 0,
                  where=synthetic_seismogram > 0, color='black', linewidth=1)
plt.fill_betweenx(np.arange(len(synthetic_seismogram)), synthetic_seismogram, 0,
                  where=synthetic_seismogram < 0, color='white', linewidth=1)
plt.ylim(len(synthetic_seismogram), 0)  # Y轴反转以便深度从上到下增加
plt.xlabel('Amplitude')
plt.ylabel('Depth')
plt.tight_layout()
plt.show()
