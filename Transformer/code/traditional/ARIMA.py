"""
__author__ = 'Cheng Yuchao'
__project__: ARIMA:自回归积分滑动平均模型：通常用于单变量时间序列数据的预测
__time__:  2023/09/30
__email__:"2477721334@qq.com"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
import itertools
import statsmodels.api as sm

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 导入数据
data = pd.read_csv('../../data/Well3_EPOR0_1.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0
data_x = data['GR']

# 确定ARIMA模型的参数(p, d, q)
# 通常需要进行模型选择和调参来找到合适的参数

# p = 1  # 自回归阶数
# d = 1  # 差分阶数
# q = 1  # 移动平均阶数

# 定义要搜索的参数范围
p = d = q = range(0, 3)

# 创建参数网格
pdq = list(itertools.product(p, d, q))

# 用AIC评估每个模型
best_aic = np.inf
best_params = None

for param in pdq:
    try:
        model = sm.tsa.ARIMA(data_x, order=param)
        results = model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = param
    except:
        continue

print(f"Best ARIMA Model: {best_params} with AIC = {best_aic}")

# 创建ARIMA模型
model = ARIMA(data_x, order=best_params)

# 拟合模型
model_fit = model.fit()

# 生成预测
forecast = model_fit.forecast(steps=100)  # 在这里预测未来10个时间点

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(data_x, label='Observed')
# 创建一个全为空值的长度为1000的Series
zero_series = pd.Series([np.nan] * len(data_x))
plt.plot(pd.concat([zero_series, forecast]), label='Forecast')
plt.xlim(5000,)
plt.legend()
plt.title('ARIMA Forecast')
plt.show()
