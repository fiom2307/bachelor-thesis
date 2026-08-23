import numpy as np
from sklearn.metrics import accuracy_score

from src.data.dataset import get_data_for_subject
from src.models.csp_lda import (
    predict_csp_lda,
    train_or_load_csp_lda,
    train_or_load_csp_lda_time_window,
    train_or_load_csp_lda_n_components,
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


def evaluate_csp_lda_time_window_for_subject(
    subject: int,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, float]:
    data = get_data_for_subject(
        subject, 
        tmin, 
        tmax,
    )

    if data is None:
        return None

    X_train, y_train, X_eval, y_eval = data

    models = train_or_load_csp_lda_time_window(
        subject,
        X_train,
        y_train,
        tmin,
        tmax,
    )

    y_pred = predict_csp_lda(
        models,
        X_eval,
    )

    accuracy = float(accuracy_score(y_eval, y_pred))

    return accuracy, y_pred


def evaluate_csp_lda_n_components_for_subject(
    subject: int,
    n_components: int,
) -> tuple[float, np.ndarray] | None:
    data = get_data_for_subject(subject)

    if data is None:
        return None

    X_train, y_train, X_eval, y_eval = data

    models = train_or_load_csp_lda_n_components(
        subject,
        X_train,
        y_train,
        n_components,
    )

    y_pred = predict_csp_lda(
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