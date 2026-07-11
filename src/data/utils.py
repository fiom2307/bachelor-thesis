import numpy as np

def get_train_labels(epochs):
    class_to_label = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3,
    }

    event_code_to_label = {
        epochs.event_id[class_name]: label
        for class_name, label in class_to_label.items()
    }

    return np.array(
        [
            event_code_to_label[event_code]
            for event_code in epochs.events[:, -1]
        ],
        dtype=np.int64,
    )