from sklearn.metrics import accuracy_score

from src.data.data_loader import (
    load_epochs, 
    load_left_right_true_labels,
    get_subject_files
)

from src.models.eegnet import (
    train_or_load_eegnet,
    predict_eegnet,
)

from src.data.preprocessing import (
    normalize_epochs,
    prepare_eegnet_input
)


def get_eegnet_for_subject(subject: int):
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

def run_eegnet_for_subject(subj):

    data = get_eegnet_for_subject(subj)

    if data is None:
        return None

    X_train, y_train, X_eval, y_eval = data

    X_train, X_eval = normalize_epochs(X_train, X_eval)

    X_train = prepare_eegnet_input(X_train)
    X_eval = prepare_eegnet_input(X_eval)

    model = train_or_load_eegnet(
        subj,
        X_train,
        y_train,
    )

    y_pred = predict_eegnet(model, X_eval)

    accuracy = accuracy_score(y_eval, y_pred)

    return accuracy