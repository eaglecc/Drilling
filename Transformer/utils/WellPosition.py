import pandas as pd
import matplotlib.pyplot as plt
import warnings
import math

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图例中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 假设文件名为wells.xlsx，sheet名为wells
df = pd.read_excel('../data/ESH well tops to estimate Cornell tops.xlsx', sheet_name='data')

selected_values = [31011161200000, 31099204460000, 31101216240000, 31109227670000, 31109227890000, 31109229980000]
filtered_df = df[df['UWI/API'].isin(selected_values)]

API = df['UWI/API']
SURFLAT = df['SURFLAT']  # 维度
SURFLON = df['SURFLON']  # 经度

plt.figure(figsize=(10, 8))
plt.scatter(SURFLON, SURFLAT, marker='o', color='blue', label='井的位置')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.title('井的位置分布图')
plt.legend()
plt.grid(True)
plt.show()


# 计算经纬度之间的直线距离
def haversine(lat1, lon1, lat2, lon2):
    # 地球半径（千米）
    R = 6371.0
    # 将角度转换为弧度
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    # 差值
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine公式计算距离
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # 计算距离（米）
    distance = R * c * 1000
    return distance


# 经度和纬度
lon1 = -76.63839
lat1 = 42.42457
lon2 = -76.63581
lat2 = 42.51203



# 计算距离
distance = haversine(lat1, lon1, lat2, lon2)
print("两点之间的距离：", distance, "米")

i = 0
