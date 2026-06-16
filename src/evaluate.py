import numpy as np
from sklearn.metrics import accuracy_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    return accuracy_score(y_true, y_pred)