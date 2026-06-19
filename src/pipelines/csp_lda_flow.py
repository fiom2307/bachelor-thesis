from sklearn.metrics import accuracy_score

from src.data.data_loader import get_subject_files, load_epochs
from src.data.data_loader import load_left_right_true_labels
from src.models.csp_lda import (
    train_csp_lda_pipeline, 
    predict_csp_lda_pipeline,
    load_csp_lda_models,
    save_csp_lda_models,
)


def get_csp_lda_data_for_subject(subject: int):
    files = get_subject_files(subject)

    if files is None:
        return None

    train_file, eval_file, mat_file = files

    X_train, y_train = load_epochs(train_file, None)

    if X_train is None:
        return None

    X_eval, _ = load_epochs(eval_file, mat_file)

    if X_eval is None:
        return None

    y_eval = load_left_right_true_labels(mat_file)

    return X_train, y_train, X_eval, y_eval


def run_csp_lda_for_subject(subject: int):
    data = get_csp_lda_data_for_subject(subject)

    if data is None:
        return None

    X_train, y_train, X_eval, y_eval = data

    saved_models = load_csp_lda_models(subject)
    force_retrain = False

    if saved_models is not None and not force_retrain:
        csp, lda = saved_models
    else:
        csp, lda = train_csp_lda_pipeline(
            X_train,
            y_train
        )

        save_csp_lda_models(
            subject,
            csp,
            lda
        )

    y_pred = predict_csp_lda_pipeline(
        csp,
        lda,
        X_eval
    )

    accuracy = accuracy_score(y_eval, y_pred)

    return accuracy