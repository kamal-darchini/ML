from sklearn.decomposition import PCA


class MyPCA:

    def __init__(self):
        self.pca = None

    def fit(self, features):
        self.pca = PCA(n_components=2)
        self.pca.fit(features)

    def transform(self, features):
        return self.pca.transform(features)