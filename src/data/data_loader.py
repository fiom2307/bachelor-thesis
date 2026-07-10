from pathlib import Path
from scipy.io import loadmat
import mne

from src.utils.paths import (
    is_eval_file
)

from src.data.utils import (
    to_binary_left_right,
    get_left_right_mask,
    filter_left_right_epochs,
    get_train_left_right_labels
)
from src.data.preprocessing import (
    pick_eeg_channels, 
    apply_bandpass_filter,
    extract_events,
    get_event_ids_for_session,
    create_epochs,
    get_epochs_data
)

def load_raw_gdf(file_path: Path):
    file_path = Path(file_path)
    raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)

    return raw

def load_true_labels_full(mat_path: Path):
    mat_path = Path(mat_path)
    labels = loadmat(mat_path)["classlabel"].flatten()

    # labels are 1, 2, 3, 4
    return labels

def load_left_right_true_labels(mat_file: Path):
    y_true_full = load_true_labels_full(mat_file)
    mask_lr = get_left_right_mask(y_true_full)
    y_true_lr = y_true_full[mask_lr]
    y_true_lr = to_binary_left_right(y_true_lr, 1)
    return y_true_lr

def load_epochs(file_path: Path, mat_path: Path):
    raw = load_raw_gdf(file_path)

    raw_eeg = pick_eeg_channels(raw)

    apply_bandpass_filter(raw_eeg)

    events, event_id = extract_events(raw)

    is_eval = is_eval_file(file_path)

    event_id_used = get_event_ids_for_session(event_id, is_eval)

    epochs = create_epochs(raw_eeg, events, event_id_used)

    epochs_data = get_epochs_data(epochs)

    if is_eval:
        X = filter_left_right_epochs(load_true_labels_full(mat_path), epochs_data)
        y = None
    else:
        X = epochs_data
        y = get_train_left_right_labels(epochs, event_id_used["left_hand"])

    return X, y