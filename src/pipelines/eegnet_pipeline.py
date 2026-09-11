import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import accuracy_score

from src.data.preprocessing import (
    normalize_epochs,
    prepare_eegnet_input,
)
from src.models.eegnet import (
    predict_eegnet,
    train_or_load_eegnet,
    train_or_load_eegnet_time_window,
)


def evaluate_eegnet_for_subject(
    subject: int,
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> tuple[float, np.ndarray]:
    """Evaluate the EEGNet ensemble and return its accuracy and predictions."""
    X_train, y_train, X_eval, y_eval = data

    X_train, X_eval = normalize_epochs(
        X_train,
        X_eval,
    )

    X_train = prepare_eegnet_input(X_train)
    X_eval = prepare_eegnet_input(X_eval)

    models = train_or_load_eegnet(
        subject,
        X_train,
        y_train,
    )

    y_pred = predict_eegnet(
        models,
        X_eval,
    )

    accuracy = float(accuracy_score(y_eval, y_pred))
    return accuracy, y_pred


def evaluate_eegnet_time_window_for_subject(
    subject: int,
    tmin: float,
    tmax: float,
) -> tuple[float, np.ndarray] | None:
    """
    Evaluate EEGNet trained from the requested temporal window.
    """
    from src.data.dataset import get_data_for_subject

    data = get_data_for_subject(
        subject,
        tmin,
        tmax,
    )

    if data is None:
        return None

    X_train, y_train, X_eval, y_eval = data

    X_train, X_eval = normalize_epochs(
        X_train,
        X_eval,
    )

    X_train = prepare_eegnet_input(X_train)
    X_eval = prepare_eegnet_input(X_eval)

    models = train_or_load_eegnet_time_window(
        subject,
        X_train,
        y_train,
        tmin,
        tmax,
    )

    y_pred = predict_eegnet(
        models,
        X_eval,
    )

    accuracy = float(accuracy_score(y_eval, y_pred))

    return accuracy, y_pred
