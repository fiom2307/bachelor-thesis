import numpy as np
from sklearn.metrics import accuracy_score

from src.models.csp_lda import (
    predict_csp_lda,
    train_or_load_csp_lda,
)


def evaluate_csp_lda_for_subject(
    subject: int,
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[float, np.ndarray]:
    """Evaluate CSP+LDA and return its accuracy and predictions."""
    X_train, y_train, X_eval, y_eval = data

    models = train_or_load_csp_lda(
        subject,
        X_train,
        y_train,
    )

    y_pred = predict_csp_lda(
        models,
        X_eval,
    )

    accuracy = float(accuracy_score(y_eval, y_pred))

    return accuracy, y_pred