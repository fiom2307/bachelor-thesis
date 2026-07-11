from pathlib import Path
from scipy.io import loadmat
import mne
import numpy as np

from src.utils.paths import (
    is_eval_file
)

from src.data.utils import (
    get_train_labels
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

    # Original labels:
    # 1 = left hand
    # 2 = right hand
    # 3 = feet
    # 4 = tongue
    #
    # Convert to:
    # 0 = left hand
    # 1 = right hand
    # 2 = feet
    # 3 = tongue
    return labels.astype(np.int64) - 1

def load_epochs(file_path: Path, mat_path: Path):
    raw = load_raw_gdf(file_path)

    raw_eeg = pick_eeg_channels(raw)

    apply_bandpass_filter(raw_eeg)

    events, event_id = extract_events(raw)

    is_eval = is_eval_file(file_path)

    event_id_used = get_event_ids_for_session(
        event_id,
        is_eval,
    )

    if event_id_used is None:
        return None, None

    epochs = create_epochs(
        raw_eeg,
        events,
        event_id_used,
    )

    X = get_epochs_data(epochs)

    if is_eval:
        if mat_path is None:
            raise ValueError(
                "mat_path is required for an evaluation GDF file."
            )

        y = load_true_labels_full(mat_path)
    else:
        y = get_train_labels(epochs)

    if len(X) != len(y):
        raise ValueError(
            f"Different number of epochs and labels: "
            f"{len(X)} epochs, {len(y)} labels."
        )

    return X, y