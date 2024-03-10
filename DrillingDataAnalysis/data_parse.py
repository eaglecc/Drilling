import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 导入数据
original_data = pd.read_csv('./data/t_original_data.csv' , header=None)
column_names = ['ID', 'created_at', 'updated_at','deleted_at','oid','pressure1',
                'pressure2','pressure3','pressure4','pressure5','pressure6','pressure7','pressure8',
                'ch1','ch2','co1','co2','speed1','distance1','start_time','end_time','v']
original_data.columns = column_names
original_data["end_time"] = pd.to_datetime(original_data["end_time"])
# 选择时间在2024-01-30及之后的所有数据
selected_data = original_data[original_data['end_time'] >= '2024-01-30']

def filterData(data):
    # 使用百分位数移除异常值
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    filtered_data = data[(data >= q25 - 1.5 * iqr) & (data <= q75 + 1.5 * iqr)]
    # 删除所有为0的值
    filtered_data = filtered_data[filtered_data != 0].reset_index(drop=False)
    return filtered_data

# 提取距离的数据
distance_data = selected_data["distance1"]
filtered_distance_data = filterData(distance_data)

# 绘制曲线图
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(filtered_distance_data["index"], filtered_distance_data["distance1"], label='Distance')  # 绘制曲线，你可以自定义标签
plt.title('测距曲线图')  # 设置标题
plt.xlabel('ID')  # 设置X轴标签
plt.ylabel('测距')  # 设置Y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形

def identify_trends(data):
    trends = []
    data_distance = data["distance1"]
    data_idx = data["index"]
    current_trend = None
    trend_start_index = 0

    for i in range(1, len(data)):
        if data_distance[i] > data_distance[i - 1]:
            if current_trend == "decreasing":
                # 连续减少大于 0.5m   且 时间间隔需大于一分钟（序号差大于5） 被认为是钻进段
                if (data_distance[trend_start_index] - data_distance[i-1] > 0.5) and (data_idx[i - 1] - data_idx[trend_start_index] >= 5):
                    trends.append((data_idx[trend_start_index], data_idx[i - 1], "decreasing"))
                    trend_start_index = i
            current_trend = "increasing"
        elif data_distance[i] < data_distance[i - 1]:
            if current_trend == "increasing":
                trends.append((data_idx[trend_start_index], data_idx[i - 1], "increasing"))
                trend_start_index = i
            current_trend = "decreasing"

    # Handle the last trend
    if trend_start_index < len(data) - 1:
        trends.append((data_idx[trend_start_index], data_idx[len(data) - 1], current_trend))

    return trends

# 识别增长段和降低段
trends = identify_trends(filtered_distance_data)

# 打印结果
for trend in trends:
    start_index, end_index, trend_type = trend
    print(f"{trend_type} trend from index {start_index} to {end_index}")