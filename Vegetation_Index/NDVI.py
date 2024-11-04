import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 读取数据
data = pd.read_csv(r'.csv', header=None) #加载数据
wavelengths = data.iloc[0, :-1]  # 波长范围
values = data.iloc[1:, :-1]  # 光谱反射率
target = data.iloc[1:, -1]  # 真实值含量

# 选择相关性计算方法
index_types = ['difference', 'ratio', 'normalized_difference', 'renormalized_difference', 'NDVI', 'EVI', 'SAVI', 'GNDVI', 'RVI', 'DVI', 'NDWI', 'MSAVI']

for index_type in index_types:
    # 计算波段指数的相关性
    correlations = np.zeros((len(wavelengths), len(wavelengths)))

    for i in range(len(wavelengths)):
        for j in range(len(wavelengths)):
            if i != j:
                if index_type == 'difference':
                    index = values.iloc[:, i] - values.iloc[:, j]
                elif index_type == 'ratio':
                    index = values.iloc[:, i] / values.iloc[:, j]
                elif index_type == 'normalized_difference':
                    index = (values.iloc[:, i] - values.iloc[:, j]) / (values.iloc[:, i] + values.iloc[:, j])
                elif index_type == 'renormalized_difference':
                    index = (values.iloc[:, i] - values.iloc[:, j]) / np.sqrt(values.iloc[:, i] + values.iloc[:, j])
                elif index_type == 'NDVI':
                    index = (values.iloc[:, i] - values.iloc[:, j]) / (values.iloc[:, i] + values.iloc[:, j])
                elif index_type == 'EVI':
                    G = 2.5
                    C1 = 6
                    C2 = 7.5
                    L = 1
                    index = G * (values.iloc[:, i] - values.iloc[:, j]) / (values.iloc[:, i] + C1 * values.iloc[:, j] - C2 * values.iloc[:, min(i+1, len(wavelengths)-1)] + L)
                elif index_type == 'SAVI':
                    L = 0.5
                    index = (values.iloc[:, i] - values.iloc[:, j]) * (1 + L) / (values.iloc[:, i] + values.iloc[:, j] + L)
                elif index_type == 'GNDVI':
                    index = (values.iloc[:, i] - values.iloc[:, j]) / (values.iloc[:, i] + values.iloc[:, j])
                elif index_type == 'RVI':
                    index = values.iloc[:, i] / values.iloc[:, j]
                elif index_type == 'DVI':
                    index = values.iloc[:, i] - values.iloc[:, j]
                elif index_type == 'NDWI':
                    index = (values.iloc[:, i] - values.iloc[:, j]) / (values.iloc[:, i] + values.iloc[:, j])
                elif index_type == 'MSAVI':
                    index = (2 * values.iloc[:, i] + 1 - np.sqrt((2 * values.iloc[:, i] + 1)**2 - 8 * (values.iloc[:, i] - values.iloc[:, j]))) / 2
                else:
                    raise ValueError("Invalid index type. Choose 'difference', 'ratio', 'normalized_difference', 'renormalized_difference', 'NDVI', 'EVI', 'SAVI', 'GNDVI', 'RVI', 'DVI', 'NDWI', 'MSAVI'.")

                # 计算指数和真实值的相关性
                correlations[i, j] = index.corr(target)

    # 找出相关系数最大的5组波长对
    flat_correlations = correlations.flatten()
    top_5_indices = np.argpartition(-np.abs(flat_correlations), 20)[:20]
    top_5_indices = top_5_indices[np.argsort(-np.abs(flat_correlations[top_5_indices]))]
    top_5_pairs = np.unravel_index(top_5_indices, correlations.shape)
    top_5_wavelengths = [(wavelengths.iloc[top_5_pairs[0][i]], wavelengths.iloc[top_5_pairs[1][i]]) for i in range(20)]
    top_5_values = [correlations[top_5_pairs[0][i], top_5_pairs[1][i]] for i in range(20)]

    print(f"Index Type: {index_type}")
    for i, (wavelength_pair, corr_value) in enumerate(zip(top_5_wavelengths, top_5_values)):
        print(f"第 {i + 1} 组相关系数最大的波长对是: x轴：{wavelength_pair[1]}nm 和 y轴：{wavelength_pair[0]}nm")
        print(f"相关系数为: {corr_value}")

    # 定义一个加深的coolwarm colormap
    colors = [
        (0.0, (40 / 255, 67 / 255, 159 / 255)),  # 深蓝色
        (0.2, 'blue'),  # 标准蓝色
        (0.5, 'white'),  # 中间是白色
        (0.7, 'red'),  # 标准红色
        (1.0, (128 / 255, 21 / 255, 25 / 255))  # 深红色
    ]
    deep_coolwarm = LinearSegmentedColormap.from_list('deep_coolwarm', colors)

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=False, cmap=deep_coolwarm, square=True, vmin=-0.85, vmax=0.85)

    # 设置X轴和Y轴的刻度标签
    xtick_labels = ytick_labels = np.arange(400, 1001, 100)  # X轴和Y轴标签一致
    xtick_positions = ytick_positions = np.linspace(0, len(wavelengths) - 1, len(xtick_labels))

    plt.xticks(xtick_positions, xtick_labels)
    plt.yticks(ytick_positions, ytick_labels)

    # 绘制从左下角开始的对角线
    plt.plot([0, len(wavelengths) - 1], [0, len(wavelengths) - 1], color='black', linewidth=1)

    plt.title(f'Correlation Heatmap of {index_type.capitalize()} Indices with Target')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Wavelength (nm)')
    plt.gca().invert_yaxis()  # 翻转Y轴使左下角为原点
    plt.show()
