import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC

from src.models.csp import apply_csp, fit_csp
from src.models.svm import train_svm
from src.utils.config import BASE_SEED, N_FOLDS
from src.utils.cross_validation import (
    average_fold_probabilities,
    get_stratified_folds,
)
from src.utils.paths import (
    get_csp_fold_model_path,
    get_svm_fold_model_path,
    get_subject_name,
)


def train_csp_svm(
    csp: CSP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> SVC:
    """
    Train an SVM classifier using features from a fitted CSP model.
    """
    X_train_csp = apply_csp(csp, X_train)

    svm = train_svm(
        X_train_csp,
        y_train,
        seed,
    )

    return svm


def predict_proba_csp_svm(
    csp: CSP,
    svm: SVC,
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict class probabilities for EEG epochs using CSP+SVM.
    """
    X_eval_csp = apply_csp(csp, X_eval)

    return svm.predict_proba(X_eval_csp)


def save_svm_fold_model(
    subject: int,
    fold: int,
    svm: SVC,
) -> None:
    """
    Save the SVM model trained for one fold.
    """
    svm_path = get_svm_fold_model_path(
        subject,
        fold,
    )

    joblib.dump(svm, svm_path)


def load_csp_svm_fold_models(
    subject: int,
    fold: int,
) -> tuple[CSP, SVC] | None:
    """
    Load the shared CSP model and SVM model for one fold.

    Returns None if either model file does not exist.
    """
    csp_path = get_csp_fold_model_path(
        subject,
        fold,
    )
    svm_path = get_svm_fold_model_path(
        subject,
        fold,
    )

    if not (
        csp_path.exists()
        and svm_path.exists()
    ):
        return None

    csp = joblib.load(csp_path)
    svm = joblib.load(svm_path)

    return csp, svm


def train_or_load_csp_svm(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> list[tuple[CSP, SVC]]:
    """
    Train or load one CSP+SVM model for each stratified fold.

    The CSP models are shared with CSP+LDA so that both classifiers
    operate on exactly the same CSP representation.
    """
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_models = load_csp_svm_fold_models(
            subject,
            fold,
        )

        if saved_models is not None:
            models.append(saved_models)
            continue

        print(
            f"Training {get_subject_name(subject)} "
            f"CSP+SVM fold {fold}/{N_FOLDS}"
        )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        csp_path = get_csp_fold_model_path(
            subject,
            fold,
        )

        if csp_path.exists():
            csp = joblib.load(csp_path)
        else:
            csp = fit_csp(
                X_tr,
                y_tr,
            )
            joblib.dump(
                csp,
                csp_path,
            )

        svm = train_csp_svm(
            csp,
            X_tr,
            y_tr,
            seed,
        )

        save_svm_fold_model(
            subject,
            fold,
            svm,
        )

        models.append(
            (
                csp,
                svm,
            )
        )

    return models


def predict_csp_svm(
    models: list[tuple[CSP, SVC]],
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict the final classes using the CSP+SVM fold ensemble.

    Each fold model predicts class probabilities for the evaluation
    epochs. The probabilities are averaged, and the class with the
    highest average probability is selected.
    """
    probabilities = []

    for csp, svm in models:
        fold_probabilities = predict_proba_csp_svm(
            csp,
            svm,
            X_eval,
        )

        probabilities.append(
            fold_probabilities
        )

    return average_fold_probabilities(
        probabilities
    )