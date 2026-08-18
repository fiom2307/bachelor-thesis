import numpy as np
from sklearn.metrics import accuracy_score

from src.models.csp_svm import (
    predict_csp_svm,
    train_or_load_csp_svm,
)


def evaluate_csp_svm_for_subject(
    subject: int,
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[float, np.ndarray]:
    """
    Evaluate CSP+SVM and return its accuracy and predictions.
    """
    X_train, y_train, X_eval, y_eval = data

    models = train_or_load_csp_svm(
        subject,
        X_train,
        y_train,
    )

    y_pred = predict_csp_svm(
        models,
        X_eval,
    )

    accuracy = float(
        accuracy_score(
            y_eval,
            y_pred,
        )
    )

    return accuracy, y_pred