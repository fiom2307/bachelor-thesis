import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.csp import fit_csp, apply_csp
from src.models.lda import train_lda, predict_lda
from src.utils.paths import CSP_LDA_MODEL_DIR


def get_csp_fold_model_path(subject: int, fold: int, base_seed: int):
    return CSP_LDA_MODEL_DIR / f"A{subject:02d}_csp_kfold_seed{base_seed}_fold{fold}.joblib"


def get_lda_fold_model_path(subject: int, fold: int, base_seed: int):
    return CSP_LDA_MODEL_DIR / f"A{subject:02d}_lda_kfold_seed{base_seed}_fold{fold}.joblib"


def train_csp_lda(X_train, y_train):
    csp = fit_csp(X_train, y_train)

    X_train_csp = apply_csp(csp, X_train)

    lda = train_lda(X_train_csp, y_train)

    return csp, lda


def predict_csp_lda(csp, lda, X_eval):
    X_eval_csp = apply_csp(csp, X_eval)

    y_pred = predict_lda(lda, X_eval_csp)

    return y_pred

def predict_proba_csp_lda(csp, lda, X_eval):
    X_eval_csp = apply_csp(csp, X_eval)

    y_proba = lda.predict_proba(X_eval_csp)

    return y_proba


def save_csp_lda_fold_models(subject: int, fold: int, base_seed: int, csp, lda):
    csp_path = get_csp_fold_model_path(subject, fold, base_seed)
    lda_path = get_lda_fold_model_path(subject, fold, base_seed)

    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def load_csp_lda_fold_models(subject: int, fold: int, base_seed: int):
    csp_path = get_csp_fold_model_path(subject, fold, base_seed)
    lda_path = get_lda_fold_model_path(subject, fold, base_seed)

    if not (csp_path.exists() and lda_path.exists()):
        return None

    csp = joblib.load(csp_path)
    lda = joblib.load(lda_path)

    return csp, lda

def train_or_load_csp_lda(subject, X_train, y_train):
    base_seed = 42
    seed = base_seed + subject  # mismo seed que EEGNet -> mismos folds exactos

    n_folds = 5

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    models = []  # lista de tuplas (csp, lda)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        saved_models = load_csp_lda_fold_models(subject, fold, base_seed)

        if saved_models is not None:
            models.append(saved_models)
            continue

        print(f"Training A{subject:02d} CSP+LDA fold {fold}/{n_folds}")

        X_tr = X_train[train_idx]
        y_tr = y_train[train_idx]
        # val_idx (20%) queda sin usar para el fit, igual que en EEGNet
        # ahí solo se usa para early stopping, acá LDA no lo necesita

        csp, lda = train_csp_lda(X_tr, y_tr)

        save_csp_lda_fold_models(subject, fold, base_seed, csp, lda)

        models.append((csp, lda))

    return models

def predict_csp_lda_ensemble(models, X_eval):
    probabilities = []

    for csp, lda in models:
        proba = predict_proba_csp_lda(csp, lda, X_eval)
        probabilities.append(proba)

    mean_probabilities = np.mean(probabilities, axis=0)
    y_pred = np.argmax(mean_probabilities, axis=1)

    return y_pred

