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
    get_csp_time_window_fold_model_path,
    get_lda_time_window_fold_model_path,
    get_csp_n_components_fold_model_path,
    get_lda_n_components_fold_model_path,
    get_csp_channel_ablation_fold_model_path,
    get_csp_roi_ablation_fold_model_path,
    get_lda_channel_ablation_fold_model_path,
    get_lda_roi_ablation_fold_model_path,
)
from src.utils.config import (
    CSP_N_COMPONENTS,
)


def train_csp_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = CSP_N_COMPONENTS,
) -> tuple[CSP, LinearDiscriminantAnalysis]:
    """
    Train a CSP+LDA model.

    CSP transforms the EEG epochs into log-power spatial features.
    LDA then learns linear decision boundaries between the motor
    imagery classes using those features.
    """
    csp = fit_csp(X_train, y_train, n_components)

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


def save_csp_lda_time_window_fold_models(
    subject: int,
    fold: int,
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    tmin: float,
    tmax: float,
) -> None:
    """
    Save CSP and LDA models trained for one fold
    and one temporal window.
    """
    csp_path = get_csp_time_window_fold_model_path(
        subject,
        fold,
        tmin,
        tmax,
    )

    lda_path = get_lda_time_window_fold_model_path(
        subject,
        fold,
        tmin,
        tmax,
    )

    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def save_csp_lda_n_components_fold_models(
    subject: int,
    fold: int,
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    n_components: int,
) -> None:
    """
    Save CSP and LDA models trained for one fold
    using a specific number of CSP components.
    """
    csp_path = get_csp_n_components_fold_model_path(
        subject,
        fold,
        n_components,
    )

    lda_path = get_lda_n_components_fold_model_path(
        subject,
        fold,
        n_components,
    )

    joblib.dump(
        csp,
        csp_path,
    )

    joblib.dump(
        lda,
        lda_path,
    )


def save_csp_lda_channel_ablation_fold_models(
    subject: int,
    fold: int,
    condition: str,
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
) -> None:
    csp_path = get_csp_channel_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )
    lda_path = get_lda_channel_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )

    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def save_csp_lda_roi_ablation_fold_models(
    subject: int,
    fold: int,
    condition: str,
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
) -> None:
    csp_path = get_csp_roi_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )
    lda_path = get_lda_roi_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )

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


def load_csp_lda_channel_ablation_fold_models(
    subject: int,
    fold: int,
    condition: str,
) -> tuple[CSP, LinearDiscriminantAnalysis] | None:
    csp_path = get_csp_channel_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )
    lda_path = get_lda_channel_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )

    if not (csp_path.exists() and lda_path.exists()):
        return None

    return joblib.load(csp_path), joblib.load(lda_path)


def load_csp_lda_roi_ablation_fold_models(
    subject: int,
    fold: int,
    condition: str,
) -> tuple[CSP, LinearDiscriminantAnalysis] | None:
    csp_path = get_csp_roi_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )
    lda_path = get_lda_roi_ablation_fold_model_path(
        subject,
        fold,
        condition,
    )

    if not (csp_path.exists() and lda_path.exists()):
        return None

    return joblib.load(csp_path), joblib.load(lda_path)


def load_csp_lda_time_window_fold_models(
    subject: int,
    fold: int,
    tmin: float,
    tmax: float,
) -> tuple[
    CSP,
    LinearDiscriminantAnalysis,
] | None:
    """
    Load CSP and LDA models saved for one fold
    and one temporal window.

    Returns None if either model file does not exist.
    """
    csp_path = get_csp_time_window_fold_model_path(
        subject,
        fold,
        tmin,
        tmax,
    )

    lda_path = get_lda_time_window_fold_model_path(
        subject,
        fold,
        tmin,
        tmax,
    )

    if not (
        csp_path.exists()
        and lda_path.exists()
    ):
        return None

    csp = joblib.load(csp_path)
    lda = joblib.load(lda_path)

    return csp, lda


def load_csp_lda_n_components_fold_models(
    subject: int,
    fold: int,
    n_components: int,
) -> tuple[
    CSP,
    LinearDiscriminantAnalysis,
] | None:
    """
    Load CSP and LDA models saved for one fold
    using a specific number of CSP components.

    Returns None if either model file does not exist.
    """
    csp_path = get_csp_n_components_fold_model_path(
        subject,
        fold,
        n_components,
    )

    lda_path = get_lda_n_components_fold_model_path(
        subject,
        fold,
        n_components,
    )

    if not (
        csp_path.exists()
        and lda_path.exists()
    ):
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


def train_or_load_csp_lda_channel_ablation(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    condition: str,
) -> list[tuple[CSP, LinearDiscriminantAnalysis]]:
    return _train_or_load_csp_lda_ablation(
        subject=subject,
        X_train=X_train,
        y_train=y_train,
        condition=condition,
        load_fold_models=load_csp_lda_channel_ablation_fold_models,
        save_fold_models=save_csp_lda_channel_ablation_fold_models,
    )


def train_or_load_csp_lda_roi_ablation(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    condition: str,
) -> list[tuple[CSP, LinearDiscriminantAnalysis]]:
    return _train_or_load_csp_lda_ablation(
        subject=subject,
        X_train=X_train,
        y_train=y_train,
        condition=condition,
        load_fold_models=load_csp_lda_roi_ablation_fold_models,
        save_fold_models=save_csp_lda_roi_ablation_fold_models,
    )


def train_or_load_csp_lda_n_components(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int,
) -> list[tuple[CSP, LinearDiscriminantAnalysis]]:
    """
    Train or load one CSP+LDA model for each stratified fold
    using a specific number of CSP components.
    """
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_models = load_csp_lda_n_components_fold_models(
            subject,
            fold,
            n_components,
        )

        if saved_models is not None:
            models.append(saved_models)
            continue

        print(
            f"Training {get_subject_name(subject)} "
            f"CSP+LDA fold {fold}/{N_FOLDS} "
            f"with {n_components} components"
        )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        csp, lda = train_csp_lda(
            X_tr,
            y_tr,
            n_components=n_components,
        )

        save_csp_lda_n_components_fold_models(
            subject,
            fold,
            csp,
            lda,
            n_components,
        )

        models.append((csp, lda))

    return models


def _train_or_load_csp_lda_ablation(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    condition: str,
    load_fold_models,
    save_fold_models,
) -> list[tuple[CSP, LinearDiscriminantAnalysis]]:
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_models = load_fold_models(
            subject,
            fold,
            condition,
        )

        if saved_models is not None:
            _validate_csp_input_shape(
                saved_models[0],
                X_train.shape[1],
                condition,
            )
            print(
                f"[LOAD] {get_subject_name(subject)} "
                f"CSP+LDA fold {fold}/{N_FOLDS} {condition}"
            )
            models.append(saved_models)
            continue

        print(
            f"[TRAIN] {get_subject_name(subject)} "
            f"CSP+LDA fold {fold}/{N_FOLDS} {condition}"
        )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        csp, lda = train_csp_lda(X_tr, y_tr)

        save_fold_models(
            subject,
            fold,
            condition,
            csp,
            lda,
        )

        models.append((csp, lda))

    return models


def _validate_csp_input_shape(
    csp: CSP,
    n_channels: int,
    condition: str,
) -> None:
    filters = getattr(
        csp,
        "filters_",
        None,
    )

    if filters is None:
        raise ValueError(
            f"Saved CSP model for {condition} is missing fitted filters."
        )

    if filters.shape[1] != n_channels:
        raise ValueError(
            f"Saved CSP model for {condition} has incompatible channel "
            f"count. Expected {n_channels}, found {filters.shape[1]}."
        )


def train_or_load_csp_lda_time_window(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tmin: float,
    tmax: float,
) -> list[tuple[CSP, LinearDiscriminantAnalysis]]:
    """
    Train or load CSP+LDA models for a specific temporal window.
    """
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_models = load_csp_lda_time_window_fold_models(
            subject,
            fold,
            tmin,
            tmax,
        )

        if saved_models is not None:
            models.append(saved_models)
            continue

        print(
            f"Training {get_subject_name(subject)} "
            f"CSP+LDA fold {fold}/{N_FOLDS} "
            f"for {tmin:.1f}-{tmax:.1f} s"
        )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        csp, lda = train_csp_lda(X_tr, y_tr)

        save_csp_lda_time_window_fold_models(
            subject,
            fold,
            csp,
            lda,
            tmin,
            tmax,
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
