import joblib
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.models.csp import apply_csp, fit_csp
from src.models.lda import train_lda
from src.utils.config import BASE_SEED, N_FOLDS
from src.utils.cross_validation import (
    average_fold_probabilities,
    get_stratified_folds,
)
from src.utils.paths import (
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_subject_name,
)


def train_csp_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[CSP, LinearDiscriminantAnalysis]:
    """
    Train a CSP+LDA model.

    CSP transforms the EEG epochs into log-power spatial features.
    LDA then learns linear decision boundaries between the motor
    imagery classes using those features.
    """
    csp = fit_csp(X_train, y_train)

    X_train_csp = apply_csp(csp, X_train)
    lda = train_lda(X_train_csp, y_train)

    return csp, lda


def predict_proba_csp_lda(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict class probabilities for EEG epochs using CSP+LDA.

    The fitted CSP model first transforms the EEG epochs into features.
    LDA then calculates the probability of each motor imagery class.
    """
    X_eval_csp = apply_csp(csp, X_eval)

    return lda.predict_proba(X_eval_csp)


def save_csp_lda_fold_models(
    subject: int,
    fold: int,
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
) -> None:
    """Save the CSP and LDA models trained for one fold."""
    csp_path = get_csp_fold_model_path(subject, fold)
    lda_path = get_lda_fold_model_path(subject, fold)

    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def load_csp_lda_fold_models(
    subject: int,
    fold: int,
) -> tuple[CSP, LinearDiscriminantAnalysis] | None:
    """
    Load the CSP and LDA models saved for one fold.

    Returns None if either of the two model files does not exist.
    """
    csp_path = get_csp_fold_model_path(subject, fold)
    lda_path = get_lda_fold_model_path(subject, fold)

    if not (csp_path.exists() and lda_path.exists()):
        return None

    csp = joblib.load(csp_path)
    lda = joblib.load(lda_path)

    return csp, lda


def train_or_load_csp_lda(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> list[tuple[CSP, LinearDiscriminantAnalysis]]:
    """
    Train or load one CSP+LDA model for each stratified fold.

    Each model is trained on the training portion of one fold. The
    validation indices are not used because CSP+LDA does not require
    early stopping. The same folds are still used as for EEGNet to
    keep the training procedure consistent.
    """
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_models = load_csp_lda_fold_models(subject, fold)

        if saved_models is not None:
            models.append(saved_models)
            continue

        print(
            f"Training {get_subject_name(subject)} "
            f"CSP+LDA fold {fold}/{N_FOLDS}"
        )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        csp, lda = train_csp_lda(X_tr, y_tr)

        save_csp_lda_fold_models(
            subject,
            fold,
            csp,
            lda,
        )

        models.append((csp, lda))

    return models


def predict_csp_lda(
    models: list[tuple[CSP, LinearDiscriminantAnalysis]],
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict the final classes using the CSP+LDA fold ensemble.

    Each fold model predicts class probabilities for the evaluation
    epochs. The probabilities are averaged, and the class with the
    highest average probability is selected.
    """
    probabilities = []

    for csp, lda in models:
        fold_probabilities = predict_proba_csp_lda(
            csp,
            lda,
            X_eval,
        )
        probabilities.append(fold_probabilities)

    return average_fold_probabilities(probabilities)