import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.utils.config import LDA_SOLVER


def train_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearDiscriminantAnalysis:
    """
    Train a Linear Discriminant Analysis classifier on the CSP features.

    LDA learns linear decision boundaries that separate the motor imagery
    classes based on the log-power features extracted by CSP.

    The SVD solver is used because it is the default, numerically stable
    option and does not require explicitly computing the covariance matrix.
    """
    lda = LinearDiscriminantAnalysis(
        solver=LDA_SOLVER,
    )

    lda.fit(X_train, y_train)

    return lda