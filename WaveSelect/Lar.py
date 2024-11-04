
from sklearn import linear_model
import numpy as np

def Lar(X, y, nums=40):
    '''
           X : 预测变量矩阵
           y ：标签
           nums : 选择的特征点的数目，默认为40
           return ：选择变量集的索引
    '''
    Lars = linear_model.Lars()
    Lars.fit(X, y)
    corflist = np.abs(Lars.coef_)

    corf = np.asarray(corflist)
    SpectrumList = corf.argsort()[-1:-(nums+1):-1]
    SpectrumList = np.sort(SpectrumList)

    return SpectrumList