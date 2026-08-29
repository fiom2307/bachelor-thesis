import joblib
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)

from src.models.fbcsp import (
    DEFAULT_FREQUENCY_BANDS,
    DEFAULT_SFREQ,
    FBCSP,
)
from src.models.lda import train_lda
from src.utils.config import (
    BASE_SEED,
    CSP_N_COMPONENTS,
    N_FOLDS,
)
from src.utils.cross_validation import (
    average_fold_probabilities,
    get_stratified_folds,
)
from src.utils.paths import (
    get_fbcsp_lda_fold_model_path,
    get_subject_name,
)


FBCSPModel = tuple[
    FBCSP,
    LinearDiscriminantAnalysis,
]


def train_fbcsp_lda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    frequency_bands: tuple[
        tuple[float, float],
        ...
    ] = DEFAULT_FREQUENCY_BANDS,
    n_components: int = CSP_N_COMPONENTS,
    sfreq: float = DEFAULT_SFREQ,
) -> FBCSPModel:
    """
    Train an FBCSP+LDA model.

    The EEG epochs are filtered into multiple frequency bands.
    A separate CSP is trained for each frequency band.

    The CSP log-power features from all bands are concatenated
    and used to train one LDA classifier.
    """
    fbcsp = FBCSP(
        frequency_bands=frequency_bands,
        n_components=n_components,
        sfreq=sfreq,
    )

    X_train_fbcsp = fbcsp.fit_transform(
        X_train,
        y_train,
    )

    lda = train_lda(
        X_train_fbcsp,
        y_train,
    )

    return (
        fbcsp,
        lda,
    )


def predict_proba_fbcsp_lda(
    fbcsp: FBCSP,
    lda: LinearDiscriminantAnalysis,
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict class probabilities using FBCSP+LDA.
    """
    X_eval_fbcsp = fbcsp.transform(
        X_eval,
    )

    return lda.predict_proba(
        X_eval_fbcsp,
    )


def save_fbcsp_lda_fold_model(
    subject: int,
    fold: int,
    fbcsp: FBCSP,
    lda: LinearDiscriminantAnalysis,
) -> None:
    """
    Save one FBCSP+LDA fold model.
    """
    model_path = get_fbcsp_lda_fold_model_path(
        subject,
        fold,
    )

    joblib.dump(
        {
            "fbcsp": fbcsp,
            "lda": lda,
        },
        model_path,
    )


def load_fbcsp_lda_fold_model(
    subject: int,
    fold: int,
) -> FBCSPModel | None:
    """
    Load one saved FBCSP+LDA fold model.

    Returns None if the model does not exist.
    """
    model_path = get_fbcsp_lda_fold_model_path(
        subject,
        fold,
    )

    if not model_path.exists():
        return None

    saved_model = joblib.load(
        model_path,
    )

    return (
        saved_model["fbcsp"],
        saved_model["lda"],
    )


def train_or_load_fbcsp_lda(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    frequency_bands: tuple[
        tuple[float, float],
        ...
    ] = DEFAULT_FREQUENCY_BANDS,
    n_components: int = CSP_N_COMPONENTS,
    sfreq: float = DEFAULT_SFREQ,
) -> list[FBCSPModel]:
    """
    Train or load one FBCSP+LDA model for every stratified fold.

    The same fold-generation procedure as CSP+LDA and EEGNet
    is used to ensure a fair comparison.
    """
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_model = load_fbcsp_lda_fold_model(
            subject,
            fold,
        )

        if saved_model is not None:
            models.append(saved_model)
            continue

        print(
            f"Training {get_subject_name(subject)} "
            f"FBCSP+LDA fold {fold}/{N_FOLDS}"
        )

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        fbcsp, lda = train_fbcsp_lda(
            X_tr,
            y_tr,
            frequency_bands=frequency_bands,
            n_components=n_components,
            sfreq=sfreq,
        )

        save_fbcsp_lda_fold_model(
            subject,
            fold,
            fbcsp,
            lda,
        )

        models.append(
            (
                fbcsp,
                lda,
            )
        )

    return models


def predict_fbcsp_lda(
    models: list[FBCSPModel],
    X_eval: np.ndarray,
) -> np.ndarray:
    """
    Predict final classes using the FBCSP+LDA fold ensemble.

    Each fold model predicts class probabilities for the
    evaluation epochs. Probabilities are averaged across folds,
    exactly as in the standard CSP+LDA pipeline.
    """
    probabilities = []

    for fbcsp, lda in models:
        fold_probabilities = (
            predict_proba_fbcsp_lda(
                fbcsp,
                lda,
                X_eval,
            )
        )

        probabilities.append(
            fold_probabilities
        )

    return average_fold_probabilities(
        probabilities
    )