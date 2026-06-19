import numpy as np

def get_left_right_mask(labels):
    return (labels == 1) | (labels == 2)

def to_binary_left_right(labels, zero_label):
    return np.where(labels == zero_label, 0, 1)

def filter_left_right_epochs(y_true_full, epochs_data):
    mask_lr = get_left_right_mask(y_true_full)

    X_lr = epochs_data[mask_lr]

    return X_lr

def get_train_left_right_labels(epochs, left_event_id):
    # (144,)
    y = epochs.events[:, -1]
    y_binary = to_binary_left_right(y, left_event_id)

    return y_binary