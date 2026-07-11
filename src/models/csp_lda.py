import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.csp import fit_csp, apply_csp
from src.models.lda import train_lda, predict_lda
from src.utils.config import BASE_SEED

from src.utils.paths import (
    get_csp_fold_model_path,
    get_lda_fold_model_path
)

def train_csp_lda(X_train, y_train):
    csp = fit_csp(X_train, y_train)

    X_train_csp = apply_csp(csp, X_train)

    lda = train_lda(X_train_csp, y_train)

    return csp, lda

def predict_proba_csp_lda(csp, lda, X_eval):
    X_eval_csp = apply_csp(csp, X_eval)

    y_proba = lda.predict_proba(X_eval_csp)

    return y_proba


def save_csp_lda_fold_models(subject: int, fold: int, csp, lda):
    csp_path = get_csp_fold_model_path(subject, fold)
    lda_path = get_lda_fold_model_path(subject, fold)

    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def load_csp_lda_fold_models(subject: int, fold: int):
    csp_path = get_csp_fold_model_path(subject, fold)
    lda_path = get_lda_fold_model_path(subject, fold)

    if not (csp_path.exists() and lda_path.exists()):
        return None

    csp = joblib.load(csp_path)
    lda = joblib.load(lda_path)

    return csp, lda

def train_or_load_csp_lda(subject, X_train, y_train):
    seed = BASE_SEED + subject

    n_folds = 5

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        saved_models = load_csp_lda_fold_models(subject, fold)

        if saved_models is not None:
            models.append(saved_models)
            continue

        print(f"Training A{subject:02d} CSP+LDA fold {fold}/{n_folds}")

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]

        csp, lda = train_csp_lda(X_tr, y_tr)

        save_csp_lda_fold_models(subject, fold, csp, lda)

        models.append((csp, lda))

    return models

def predict_csp_lda(models, X_eval):
    probabilities = []

    for csp, lda in models:
        proba = predict_proba_csp_lda(csp, lda, X_eval)
        probabilities.append(proba)

    mean_probabilities = np.mean(probabilities, axis=0)
    y_pred = np.argmax(mean_probabilities, axis=1)

    return y_pred

