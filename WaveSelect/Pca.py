

from sklearn.decomposition import PCA

def Pca(X, nums=20):
    """
       :param X: raw spectrum data, shape (n_samples, n_features)
       :param nums: Number of principal components retained
       :return: X_reduction：Spectral data after dimensionality reduction
    """
    pca = PCA(n_components=nums)  # 保留的特征数码
    pca.fit(X)
    X_reduction = pca.transform(X)

    return X_reduction
