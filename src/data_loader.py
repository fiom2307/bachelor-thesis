import mne
import numpy as np
from scipy.io import loadmat
from pathlib import Path

def to_binary_left_right(labels, left_label):
    return np.where(labels == left_label, 0, 1)

def load_true_labels_full(mat_path: Path) -> np.ndarray:
    mat_path = Path(mat_path)
    labels = loadmat(mat_path)["classlabel"].flatten()

    # labels are 1, 2, 3, 4
    return labels

def load_left_right_true_labels(mat_file: Path):
    y_true_full = load_true_labels_full(mat_file)
    mask_lr = (y_true_full == 1) | (y_true_full == 2)
    y_true_lr = y_true_full[mask_lr]
    y_true_lr = to_binary_left_right(y_true_lr, 1)

    return y_true_lr

def load_epochs(file_path: Path, mat_path: Path):
    file_path = Path(file_path)
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

    # First 22 channels are EEG, last 3 are EOG
    raw_eeg = raw.copy().pick(raw.ch_names[:22])

    # Apply a band-pass filter from 8 Hz to 30 Hz
    raw_eeg.filter(l_freq=8, h_freq=30, verbose=False)

    # events: [sample_position = 98241, previous_value = 0, event_code = 7]
    # event_id: {"769": 7, "770": 8, ...}
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # T or E
    file_stem = file_path.stem
    session = file_stem[3]
    is_eval = session.upper() == "E"

    if is_eval:
        if "783" not in event_id:
            print(f"[{file_stem}] No cue event 783 found.")
            return None, None

        event_id_used = {"cue": event_id["783"]}
    else:
        if "769" not in event_id or "770" not in event_id:
            print(f"[{file_stem}] No cue event 769/770 found.")
            return None, None

        event_id_used = {
            "left_hand":  event_id["769"],
            "right_hand": event_id["770"],
        }

    #   BCI competition: figure 2
    #  -|---|---|---|---|-
    #   2   3   4   5   6s
    #   cue--motor imagery
    #   |------4s-------|
    epochs = mne.Epochs(
        raw_eeg,
        events,
        event_id=event_id_used,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        verbose=False
    )

    # (n_trials = 144, 22, n_times = 1001 -> 4s = 1000)
    X = epochs.get_data()

    if is_eval:
        y_binary = None

        y_true_full = load_true_labels_full(mat_path)
        mask_lr = (y_true_full == 1) | (y_true_full == 2)
        X = X[mask_lr]
    else:
        # (144,)
        y = epochs.events[:, -1]
        y_binary = to_binary_left_right(y, event_id_used["left_hand"])

    return X, y_binary

