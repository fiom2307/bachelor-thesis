import numpy as np
from mne import Epochs


CLASS_NAME_TO_LABEL = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}


def get_train_labels(epochs: Epochs) -> np.ndarray:
    """
    Convert the training epoch event codes into zero-based class labels.

    The returned labels use the following mapping:
    0 = left hand, 1 = right hand, 2 = feet, 3 = tongue.
    """
    event_code_to_label = {
        epochs.event_id[class_name]: label
        for class_name, label in CLASS_NAME_TO_LABEL.items()
    }

    return np.array(
        [
            event_code_to_label[event_code]
            for event_code in epochs.events[:, -1]
        ],
        dtype=np.int64,
    )