import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle

#数据处理
def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    return data

#使用pickle工具打包
def pack_data_as_pickle(data, output_file_path):
    with open(output_file_path, 'wb') as file:
        pickle.dump(data, file)

#读取pickle包
def read_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# # 读取 CSV 文件
# csv_file_path = 'data/training_data.csv'
# facies_data = read_csv_file(csv_file_path)
#
# # 打包为 pickle 文件
# pickle_file_path = 'data/training_data.pickle'
# pack_data_as_pickle(facies_data, pickle_file_path)
#
# # 读取 CSV 文件
# csv_file_path = 'data/facies_vectors.csv'
# facies_data = read_csv_file(csv_file_path)
#
# # 打包为 pickle 文件
# pickle_file_path = 'data/facies_vectors.pickle'
# pack_data_as_pickle(facies_data, pickle_file_path)

# 读取 pickle 文件
pickle_file_path = 'data/facies_data.pickle'
facies_data = read_pickle_file(pickle_file_path)
i = 10