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