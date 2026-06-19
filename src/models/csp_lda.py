import joblib

from src.models.csp import fit_csp, apply_csp
from src.models.lda import train_lda, predict_lda
from src.utils.paths import CSP_LDA_MODEL_DIR


def get_csp_model_path(subject: int):
    return CSP_LDA_MODEL_DIR / f"A{subject:02d}_csp.joblib"


def get_lda_model_path(subject: int):
    return CSP_LDA_MODEL_DIR / f"A{subject:02d}_lda.joblib"


def train_csp_lda_pipeline(X_train, y_train):
    csp = fit_csp(X_train, y_train)

    X_train_csp = apply_csp(csp, X_train)

    lda = train_lda(X_train_csp, y_train)

    return csp, lda


def predict_csp_lda_pipeline(csp, lda, X_eval):
    X_eval_csp = apply_csp(csp, X_eval)

    y_pred = predict_lda(lda, X_eval_csp)

    return y_pred


def save_csp_lda_models(subject: int, csp, lda):
    csp_path = get_csp_model_path(subject)
    lda_path = get_lda_model_path(subject)

    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)

    print(f"Saved CSP+LDA models for A{subject:02d}")


def load_csp_lda_models(subject: int):
    csp_path = get_csp_model_path(subject)
    lda_path = get_lda_model_path(subject)

    if not (csp_path.exists() and lda_path.exists()):
        return None

    csp = joblib.load(csp_path)
    lda = joblib.load(lda_path)

    print(f"Loaded saved CSP+LDA models for A{subject:02d}")

    return csp, lda