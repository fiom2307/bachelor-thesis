import numpy as np
from sklearn.metrics import accuracy_score

from src.models.fbcsp_lda import (
    predict_fbcsp_lda,
    train_or_load_fbcsp_lda,
)


def evaluate_fbcsp_lda_for_subject(
    subject: int,
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[
    float,
    np.ndarray,
]:
    """
    Evaluate FBCSP+LDA for one subject.

    Returns
    -------
    accuracy
        Accuracy on the evaluation session.

    y_pred
        Ensemble predictions for all evaluation trials.
    """
    X_train, y_train, X_eval, y_eval = data

    models = train_or_load_fbcsp_lda(
        subject,
        X_train,
        y_train,
    )

    y_pred = predict_fbcsp_lda(
        models,
        X_eval,
    )

    accuracy = float(
        accuracy_score(
            y_eval,
            y_pred,
        )
    )

    return (
        accuracy,
        np.asarray(y_pred),
    )