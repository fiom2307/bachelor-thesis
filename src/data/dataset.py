import numpy as np

from src.data.data_loader import load_epochs
from src.utils.paths import get_subject_files


SubjectData = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def get_data_for_subject(
    subject: int,
) -> SubjectData | None:
    """
    Load the preprocessed training and evaluation data for one subject.

    Training labels are obtained from the GDF event annotations, while
    evaluation labels are loaded from the corresponding MATLAB file.

    Returns:
        A tuple containing X_train, y_train, X_eval, and y_eval.
        Returns None if the subject files or preprocessed EEG epochs
        are unavailable.
    """
    files = get_subject_files(subject)

    if files is None:
        return None

    train_file, eval_file, mat_file = files

    X_train, y_train = load_epochs(train_file, None)

    if X_train is None or y_train is None:
        return None

    X_eval, y_eval = load_epochs(eval_file, mat_file)

    if X_eval is None or y_eval is None:
        return None

    return X_train, y_train, X_eval, y_eval