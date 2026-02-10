import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. CORRELATION THRESHOLD (Redundancy Remover)
# ==========================================
class CorrelationThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
        else:
            corr_matrix = pd.DataFrame(X).corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.to_drop_)
        return X.drop(columns=self.to_drop_)

