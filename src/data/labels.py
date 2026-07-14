import numpy as np
from mne import Epochs


CLASS_NAME_TO_LABEL = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}

# (0, 1, 2, 3)
CLASS_LABELS = tuple(
    CLASS_NAME_TO_LABEL.values()
)

# ("Left hand", "Right hand", "Feet", "Tongue")
CLASS_NAMES = tuple(
    class_name.replace("_", " ").capitalize()
    for class_name in CLASS_NAME_TO_LABEL
)


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