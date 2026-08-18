import numpy as np
from sklearn.svm import SVC


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> SVC:
    """
    Train an SVM classifier on CSP features.
    """
    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=seed,
    )

    svm.fit(X_train, y_train)

    return svm