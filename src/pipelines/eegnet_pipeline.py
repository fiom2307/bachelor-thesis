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


def run_eegnet_for_subject(
    subject: int,
    data: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
) -> float:
    """Train or load the EEGNet ensemble and evaluate it for one subject."""
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

    # rng = np.random.default_rng(42)
    # background_idx = rng.choice(len(X_train), size=50, replace=False)

    # background = X_train[background_idx].astype(np.float32)

    # -------- choose class to explain --------
    # class_index = 0  # 0 = left hand, 1 = right hand
    # class_name = "Left hand"

    # Predict labels for evaluation trials
    # y_pred_probs = model.predict(X_eval)
    # y_pred = np.argmax(y_pred_probs, axis=1)

    # Use only correctly classified trials of the selected class
    # correct_class_mask = (y_eval == class_index) & (y_pred == class_index)

    # X_explain = X_eval[correct_class_mask].astype(np.float32)

    # print(f"Correct {class_name} trials:", X_explain.shape)

    # -------- compute SHAP --------
    # explainer = shap.GradientExplainer(model, background)

    # shap_values = explainer.shap_values(X_explain)

    # print("SHAP computed")
    # print(type(shap_values))
    # print("shap_values shape:", shap_values.shape)

    # Take SHAP values for the selected output class
    # sv = shap_values[..., class_index]

    # print("sv shape before squeeze:", sv.shape)

    # sv = sv.squeeze(-1)

    # print("sv shape after squeeze:", sv.shape)

    # channel_names = [
    #     "Fz",
    #     "FC3", "FC1", "FCz", "FC2", "FC4",
    #     "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    #     "CP3", "CP1", "CPz", "CP2", "CP4",
    #     "P1", "Pz", "P2", "POz"
    # ]

    # channel_importance = np.mean(np.abs(sv), axis=(0, 2))

    # print("channel_importance shape:", channel_importance.shape)

    # plt.figure(figsize=(10, 4))
    # plt.bar(channel_names, channel_importance)
    # plt.xticks(rotation=90)
    # plt.ylabel("Mean absolute SHAP value")
    # plt.title(f"EEGNet SHAP Channel Importance - Correct {class_name}")
    # plt.tight_layout()
    # plt.show()

    y_pred = predict_eegnet(
        models,
        X_eval,
    )

    return float(accuracy_score(y_eval, y_pred))