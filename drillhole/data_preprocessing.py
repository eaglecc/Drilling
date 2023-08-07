import pickle
import pandas as pd

class ExcelPickleReader:
    def __init__(self, pkl_file_path):
        self.pkl_file_path = pkl_file_path
        self.pkl_data = None

    def load_data(self):
        # 加载pkl包文件
        with open(self.pkl_file_path, 'rb') as file:
            self.pkl_data = pickle.load(file)

    def get_sheet_data(self, sheet_name):
        if self.pkl_data is None:
            self.load_data()

        if sheet_name in self.pkl_data:
            return self.pkl_data[sheet_name]
        else:
            print(f"Sheet '{sheet_name}' not found in the pkl package.")
            return None

    def print_sheet_data(self, sheet_name):
        sheet_data = self.get_sheet_data(sheet_name)
        if sheet_data is not None:
            print(f"Sheet: {sheet_name}")
            print(sheet_data.head())  # 打印前几行数据
            print("-------------------------")

# 加载数据
pkl_file_path = '../data/DrillHoleData28.pkl'
excel_pickle_reader = ExcelPickleReader(pkl_file_path)

hole = []
for i in range(28):
    hole.append(excel_pickle_reader.get_sheet_data(str(i+1)))

# print(hole)
# print(type(hole))





