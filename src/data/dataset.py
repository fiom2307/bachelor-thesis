from src.data.data_loader import (
    load_epochs,
    load_left_right_true_labels,
)

from src.utils.paths import (
    get_subject_files
)


def get_data_for_subject(subject: int):
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