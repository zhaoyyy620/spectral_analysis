import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Clustering.Cluster import Cluster
from Regression.Rgs import QuantitativeAnalysis
from Classification.Cls import QualitativeAnalysis

#光谱聚类分析
def SpectralClusterAnalysis(data, label, ProcessMethods, FslecetedMethods, ClusterMethods):
    """
     :param data: shape (n_samples, n_features), 光谱数据
     :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
     :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
     :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
     :param ClusterMethods : string, 聚类的方法，提供Kmeans聚类、FCM聚类
     :return: Clusterlabels: 返回的隶属矩阵
     """

    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, _ = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    Clusterlabels = Cluster(ClusterMethods, FeatrueData)
    #ClusterPlot(data, Clusterlabels)
    return Clusterlabels

# 光谱定量分析
def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):

    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定量分析模型, 包括ANN、PLS、SVR、ELM、CNN、SAE等，后续会不断补充完整
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
    """
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    Rmse, R2, Mae = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test )
    return Rmse, R2, Mae

# 光谱定性分析
def SpectralQualitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):

    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法, 提供UVE、SPA、Lars、Cars、Pca
    :param SetSplitMethods : string, 划分数据集的方法, 提供随机划分、KS划分、SPXY划分
    :param model : string, 定性分析模型, 包括ANN、PLS_DA、SVM、RF、CNN、SAE等，后续会不断补充完整
    :return: acc： float, 分类准确率
    """

    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    acc = QualitativeAnalysis(model, X_train, X_test, y_train, y_test )

    return acc





if __name__ == '__main__':

    # 载入原始数据并可视化
    # Load and visualize raw data
    data, label = LoadNirtest('Rgs')
    plt.figure(figsize=(10, 9))
    for i in range(data.shape[0]):
        plt.plot(data[i, :])

    plt.xlabel("Wavenumber(nm)")
    plt.ylabel("Reflectance")
    plt.title("The spectrum of the raw for dataset", fontweight="semibold", fontsize='large')
    plt.legend()
    plt.show()

    # 光谱预处理并可视化
    # Spectral preprocessing and visualization
    " 可选如下：None ， MMS , SS ，CT， SNV， MA， SG，MSC，FD1， FD2，DT，WVAE "

    method = "SNV"
    Preprocessingdata = Preprocessing(method, data)

    " 这里可以添加第二种预处理 "
    # method = "MSC"
    # Preprocessingdata = Preprocessing(method, Preprocessingdata)


    # 绘制预处理后归一化的光谱曲线
    # Plot the normalized spectral curves after preprocessing
    plt.figure(figsize=(10, 9))
    for i, spectrum in enumerate(Preprocessingdata):
        plt.plot(np.linspace(400, 1000, num=Preprocessingdata.shape[1]), spectrum, linewidth=1.5)  # 400-1000高光谱波段范围内作图
    plt.xlabel("Wavenumber(nm)", fontweight="semibold")
    plt.ylabel("Reflectance", fontweight="semibold")
    plt.title("The spectrum of the raw for dataset", fontweight="semibold", fontsize='large')
    # plt.xlim(375, 1000)  # 设置x轴坐标范围
    # plt.ylim(0.00, 1)  # 设置y轴坐标范围
    # 加强坐标轴数字的清晰度和突出度
    plt.tick_params(axis='both', which='major', labelsize=14, width=1, length=6)
    plt.tick_params(axis='both', which='minor', labelsize=12, width=1, length=3)
    plt.legend()
    plt.show()


    # 波长特征筛选并可视化
    # Wavelength feature selection and visualization.
    """可选如下："None"  "Cars" "Lars"  "Uve" "Spa" "GA"  "Pca"  """

    data = Preprocessingdata
    method = "None"
    SpectruSelected, y = SpctrumFeatureSelcet(method, data, label)
    print("全光谱数据维度")
    print(len(data[0,:]))
    print("经过{}波长筛选后的数据维度".format(method))
    print(len(SpectruSelected[0, :]))



    # 划分数据集
    # Split the dataset
    """可选如下："spxy"  "random" "ks"   """
    X_train, X_test, y_train, y_test = SetSplit('spxy', SpectruSelected, y, 0.2, 123)



    # 回归建模分析
    # Regression modeling and analysis.
    """ 机器学习模型可选如下："PLSR"  "ANN"  "SVR"  "ELM"  "RF" """
    """ 深度学习模型可选如下： ConNet、AlexNet、DeepSpectra、LPCNet """


    methods= 'CNN'
    print(methods)
    RMSE, R2, MAE  = QuantitativeAnalysis("CNN", X_train, X_test, y_train, y_test)
    print("The RMSE:{} R2:{}, MAE:{} of result!".format(RMSE, R2, MAE))