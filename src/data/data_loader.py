from pathlib import Path

import mne
from mne.io import BaseRaw
import numpy as np
from scipy.io import loadmat

from src.data.labels import get_train_labels
from src.data.preprocessing import (
    apply_bandpass_filter,
    create_epochs,
    extract_events,
    get_epochs_data,
    get_event_ids_for_session,
    pick_eeg_channels,
)
from src.utils.paths import is_eval_file


def load_raw_gdf(file_path: str | Path) -> BaseRaw:
    """
    Load a continuous EEG recording from a GDF file.

    The returned MNE Raw object contains the continuous channel signals,
    sampling information, channel metadata, and event annotations.
    The recording has not yet been divided into epochs.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"GDF file not found: {file_path}")

    return mne.io.read_raw_gdf(
        file_path,
        preload=True,
        verbose=False,
    )


def load_true_labels(mat_path: str | Path) -> np.ndarray:
    """
    Load the true class labels from a MATLAB evaluation file.

    The original dataset uses labels from 1 to 4:
    1 = left hand, 2 = right hand, 3 = feet, 4 = tongue.

    The labels are converted to zero-based values from 0 to 3
    so they can be used directly by the classification models.
    """
    mat_path = Path(mat_path)

    if not mat_path.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    
    mat_data = loadmat(mat_path)
    
    if "classlabel" not in mat_data:
        raise KeyError(f"'classlabel' not found in MAT file: {mat_path}")

    labels = mat_data["classlabel"].flatten()

    return labels.astype(np.int64) - 1


def load_epochs(
    file_path: str | Path, 
    mat_path: str | Path | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load and preprocess EEG trials from a GDF recording.

    The continuous EEG recording is loaded, the 22 EEG channels are selected,
    and an 8–30 Hz band-pass filter is applied. The recording is then divided
    into epochs from 0.5 to 4.0 seconds relative to each motor imagery cue.

    Training labels are obtained from the GDF event annotations.
    Evaluation labels are loaded from the corresponding MATLAB file.

    Returns:
        A tuple containing the EEG epochs and their class labels.
        If the required cue events are not found, returns (None, None).
    """
    file_path = Path(file_path)

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
                f"A MAT label file is required for evaluation file: {file_path}"
            )

        y = load_true_labels(mat_path)
    else:
        y = get_train_labels(epochs)

    if len(X) != len(y):
        raise ValueError(
            f"Different number of epochs and labels: "
            f"{len(X)} epochs, {len(y)} labels."
        )

    return X, y