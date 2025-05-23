from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class LabelEncoderPipelineFriendly(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        self.le.fit(X)
        return self

    def transform(self, X):
        return self.le.transform(X).reshape(-1, 1)
