import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def train_lda(X_train: np.ndarray, y_train: np.ndarray) -> LinearDiscriminantAnalysis:
    # to search why svd and not lsqr, eigen
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X_train, y_train)
    return lda


def predict(lda: LinearDiscriminantAnalysis, X: np.ndarray) -> np.ndarray:
    return lda.predict(X)