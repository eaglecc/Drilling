from data_preprocessing import  ExcelPickleReader
import pandas as pd
import  matplotlib.pyplot as plt

# 1. 加载数据
pkl_file_path = '../data/DrillHoleData28.pkl'
excel_pickle_reader = ExcelPickleReader(pkl_file_path)

hole = []
for i in range(28):
    hole.append(excel_pickle_reader.get_sheet_data(str(i+1)))

# 2. 删除空值
columns_to_delete = ['钻进速度', '冲击压力', '回退压力','油压/水流量','进水口压力','泵后压力','旋转压力.1']  # 要删除的列名列表
# 循环遍历每个DataFrame，删除指定的列
for df in hole:
    df.drop(columns=columns_to_delete, inplace=True)

# 3. 零值处理
zero_list = []
for i, df in enumerate(hole):
    zero_data = df[(df == 0).any(axis=1)]
    # 旋转压力列0较多，剔除该列然后筛选
    zero_data.drop(columns='旋转压力', inplace=True)
    # 剔除旋转压力列后其余列中存在0的行
    zero_data = zero_data[(zero_data == 0).any(axis=1)]
    #print(zero_data.index)
    zero_list.append(zero_data)
    # 删除索引位置
    hole[i] = hole[i].drop(zero_data.index)

# 3.1 zero_list 存储的是存在0的行，将这些数据使用插值的方式进行填充

# 4. 绘制曲线
# for i,df in enumerate(hole):
#     plt.plot(df['TimeString'],df['深度'])
#     plt.plot(df['TimeString'],df['推进压力'])
#     plt.plot(df['TimeString'],df['缓冲压力'])
#     plt.title(f"hole:{i}")
#     plt.show()

# 5. 索引处理为时间
for i,df in enumerate(hole):
    # 将时间字符串列'TimeString'转换为时间戳并设置为索引
    df['TimeString'] = pd.to_datetime(df['TimeString'])
    df.set_index('TimeString', inplace=True)
    hole[i] = df
# print(hole[0]['2018/7/2 20:29'])


# 6. 使用resample()方法按分钟重采样并计算每分钟内的均值。
hole_minutes = []
for i,df in enumerate(hole):
    # 按分钟重采样并计算每分钟内的均值
    df_resampled = df.resample('T').mean()
    hole_minutes.append(df_resampled)
    #print(df_resampled)

# 7. 绘制分钟曲线
# for i,df in enumerate(hole_minutes):
#     plt.rc('font',family='FangSong',weight='bold')
#     plt.plot(df.index,df['深度'],label='深度')
#     plt.plot(df.index,df['推进压力'],label='推进压力')
#     plt.plot(df.index,df['缓冲压力'],label='缓冲压力')
#     plt.title(f"钻孔:{i}")
#     plt.legend()
#     plt.show()

# 8. 移动平均修匀  rolling()方法
window_size = 3  # 移动平均窗口大小
for i,df in enumerate(hole_minutes):
    plt.rc('font',family='FangSong',weight='bold')

    plt.plot(df.index,df['深度'],label='深度')
    df['MovingAverage1'] = df['深度'].rolling(window=window_size).mean()
    plt.plot(df.index, df['MovingAverage1'], label=f'深度移动平均（窗口大小={window_size}）')

    plt.plot(df.index,df['推进压力'],label='推进压力')
    df['MovingAverage2'] = df['推进压力'].rolling(window=window_size).mean()
    plt.plot(df.index, df['MovingAverage2'], label=f'推进压力移动平均（窗口大小={window_size}）')

    plt.plot(df.index,df['缓冲压力'],label='缓冲压力')
    df['MovingAverage3'] = df['缓冲压力'].rolling(window=window_size).mean()
    plt.plot(df.index, df['MovingAverage3'], label=f'缓冲压力移动平均（窗口大小={window_size}）')

    plt.title(f"钻孔:{i}")
    plt.legend()
    plt.show()

# 9.移动中位数修匀  平滑处理用rolling()方法 计算中位数时调用median()函数
# 计算移动中位数
window_size = 3  # 移动窗口大小
for i,df in enumerate(hole_minutes):
    plt.rc('font',family='FangSong',weight='bold')

    plt.plot(df.index,df['深度'],label='深度')
    df['MovingMedian1'] = df['深度'].rolling(window=window_size).median()
    plt.plot(df.index, df['MovingMedian1'], label=f'深度移动中位数（窗口大小={window_size}）')

    plt.plot(df.index,df['推进压力'],label='推进压力')
    df['MovingMedian2'] = df['推进压力'].rolling(window=window_size).median()
    plt.plot(df.index, df['MovingMedian2'], label=f'推进压力移动中位数（窗口大小={window_size}）')

    plt.plot(df.index,df['缓冲压力'],label='缓冲压力')
    df['MovingMedian3'] = df['缓冲压力'].rolling(window=window_size).median()
    plt.plot(df.index, df['MovingMedian3'], label=f'缓冲压力移动中位数（窗口大小={window_size}）')

    plt.title(f"钻孔:{i}")
    plt.legend()
    plt.show()

i= 0