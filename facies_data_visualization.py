import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

'''
岩性数据可视化：展示五条测井曲线以及对应的岩性色块
调用方式：facies = FaciesVisualization('SHRIMPLIN') # 绘制矿井名称为SHRIMPLIN的曲线
'''


class FaciesVisualization:
    # 读取pickle包
    def read_pickle_file(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    # 在最后一列添加岩性标签（根据第一列对应）
    def label_facies(self, row, labels):
        return labels[row['Facies'] - 1]

    def __init__(self, wellName):
        pickle_file_path = 'data/facies_vectors.pickle'
        training_data = self.read_pickle_file(pickle_file_path)
        # The 'Well Name' and 'Formation' columns can be turned into a categorical data type.
        # 这三行代码涉及对 Pandas 数据框中的某些列进行数据类型转换和唯一值的提取。
        training_data['Well Name'] = training_data['Well Name'].astype('category')
        training_data['Formation'] = training_data['Formation'].astype('category')
        training_data['Well Name'].unique()

        # 在绘制井数据之前，定义一个颜色图，以便所有图中的相都用一致的颜色表示
        facies_colors = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00',
                         '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

        facies_labels = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS',
                         'WS', 'D', 'PS', 'BS']
        # facies_color_map is a dictionary that maps facies labels to their respective colors
        facies_color_map = {}
        for ind, label in enumerate(facies_labels):
            facies_color_map[label] = facies_colors[ind]

        training_data.loc[:, 'FaciesLabels'] = training_data.apply(lambda row: self.label_facies(row, facies_labels),
                                                                   axis=1)

        # 缺失值处理：数据集中PE列数据存在大量缺失，查看计数值，大多数值有4149个有效值，除了PE有3232个有效值。在本教程中，我们将删除没有有效PE条目的特征向量。
        PE_mask = training_data['PE'].notnull().values
        training_data = training_data[PE_mask]

        # 提取出了名称为"NOLAN"的井的数据，存储在blind变量中
        blind = training_data[training_data['Well Name'] == 'NOLAN']
        training_data = training_data[training_data['Well Name'] != 'NOLAN']

        # 调用绘制函数
        self.make_facies_log_plot(training_data[training_data['Well Name'] == wellName], facies_colors)

    # 将测井曲线绘制代码放入函数中，可以很容易地绘制多口井的测井曲线，
    # 并且当我们将相分类模型应用于其他井时，可以重用这些代码来查看结果。
    # 该函数被编写为以颜色和相标签列表作为参数。
    # 然后给出了SHRIMPLIN井的测井曲线。
    def make_facies_log_plot(self, logs, facies_colors):
        # make sure logs are sorted by depth
        logs = logs.sort_values(by='Depth')
        # 创建颜色映射（colormap）：根据提供的 facies_colors 列表创建一个颜色映射对象（cmap_facies），用于将岩相的标签映射到相应的颜色。
        cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')

        ztop = logs.Depth.min()
        zbot = logs.Depth.max()
        # 创建一个表示岩相（facies）的矩阵 cluster
        cluster = np.repeat(np.expand_dims(logs['Facies'].values, 1), 100, 1)

        f, ax = plt.subplots(nrows=1, ncols=6, figsize=(12, 6))
        ax[0].plot(logs.GR, logs.Depth, '-g')
        ax[1].plot(logs.ILD_log10, logs.Depth, '-')
        ax[2].plot(logs.DeltaPHI, logs.Depth, '-', color='0.40')
        ax[3].plot(logs.PHIND, logs.Depth, '-', color='r')
        ax[4].plot(logs.PE, logs.Depth, '-', color='black')
        # 将矩阵 cluster 映射到颜色映射 cmap_facies 中的不同颜色，并在第六个子图上显示出来
        im = ax[5].imshow(cluster, interpolation='none', aspect='auto',
                          cmap=cmap_facies, vmin=1, vmax=9)
        # cmap=cmap_facies：设置图像的颜色映射，即岩相的颜色映射。cmap_facies 是之前定义的颜色映射对象。

        divider = make_axes_locatable(ax[5])
        cax = divider.append_axes("right", size="20%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label((5 * ' ').join([' SS ', 'CSiS', 'FSiS', 'SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS ']))
        cbar.set_ticks(range(0, 1))
        cbar.set_ticklabels('')

        for i in range(len(ax) - 1):
            ax[i].set_ylim(ztop, zbot)  # 设置子图的纵轴（y轴）范围，即设置深度的上限和下限。
            ax[i].invert_yaxis()  # 将纵轴（y轴）进行反转，使深度从上到下递增。
            ax[i].grid()  # 在子图中显示网格线，以增加可读性。
            ax[i].locator_params(axis='x', nbins=3)  # 设置x轴刻度的数量为3，使得横轴的刻度数量适中。

        ax[0].set_xlabel("GR")
        ax[0].set_xlim(logs.GR.min(), logs.GR.max())  # 设置横轴范围，范围由 logs.GR 列的最小值和最大值确定。
        ax[1].set_xlabel("ILD_log10")
        ax[1].set_xlim(logs.ILD_log10.min(), logs.ILD_log10.max())
        ax[2].set_xlabel("DeltaPHI")
        ax[2].set_xlim(logs.DeltaPHI.min(), logs.DeltaPHI.max())
        ax[3].set_xlabel("PHIND")
        ax[3].set_xlim(logs.PHIND.min(), logs.PHIND.max())
        ax[4].set_xlabel("PE")
        ax[4].set_xlim(logs.PE.min(), logs.PE.max())
        ax[5].set_xlabel('Facies')

        # 参数设置为空列表，隐藏了第2、3、4、5子图的纵轴刻度标签。
        ax[1].set_yticklabels([])
        ax[2].set_yticklabels([])
        ax[3].set_yticklabels([])
        ax[4].set_yticklabels([])
        ax[5].set_yticklabels([])
        ax[5].set_xticklabels([])
        f.suptitle('Well: %s' % logs.iloc[0]['Well Name'], fontsize=14, y=0.94)
        plt.show()


# 绘制矿井名称为SHRIMPLIN的曲线
facies = FaciesVisualization('SHRIMPLIN')
