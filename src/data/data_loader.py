from pathlib import Path

import mne
from mne.io import BaseRaw
import numpy as np
from scipy.io import loadmat

from src.data.labels import get_train_labels
from src.data.preprocessing import (
    apply_bandpass_filter,
    apply_car,
    create_epochs,
    extract_events,
    get_epochs_data,
    get_event_ids_for_session,
    pick_eeg_channels,
    set_bci_2a_montage,
    select_artifact_free_cue_events,
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
    Load and preprocess artifact-free EEG trials from a GDF recording.

    The continuous EEG recording is loaded, the 22 EEG channels are selected,
    and an 8–30 Hz band-pass filter is applied. Trials marked with GDF event
    1023 are removed. The remaining recording is divided into epochs from
    0.5 to 4.0 seconds relative to each motor-imagery cue, and common average
    reference (CAR) is applied.

    Training labels are obtained from the retained GDF cue events.
    Evaluation labels are loaded from the MATLAB file and filtered using
    the same artifact-free trial mask.

    Returns:
        A tuple containing the preprocessed EEG epochs and their class labels.
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

    clean_cue_events, clean_trial_mask = (
        select_artifact_free_cue_events(
            events=events,
            event_id=event_id,
            event_id_used=event_id_used,
        )
    )

    epochs = create_epochs(
        raw_eeg,
        clean_cue_events,
        event_id_used,
    )

    X = get_epochs_data(epochs)
    X = apply_car(X)

    if is_eval:
        if mat_path is None:
            raise ValueError(
                f"A MAT label file is required for evaluation file: {file_path}"
            )

        all_labels = load_true_labels(mat_path)

        if len(all_labels) != len(clean_trial_mask):
            raise ValueError(
                "The number of evaluation labels does not match "
                "the number of motor-imagery cue events: "
                f"{len(all_labels)} labels and "
                f"{len(clean_trial_mask)} cues."
            )

        y = all_labels[clean_trial_mask]
    else:
        y = get_train_labels(epochs)

    if len(X) != len(y):
        raise ValueError(
            f"Different number of epochs and labels: "
            f"{len(X)} epochs, {len(y)} labels."
        )

    return X, y


def load_erd_epochs(
    file_path: str | Path,
    mat_path: str | Path,
) -> tuple[mne.Epochs | None, np.ndarray | None]:
    """
    Load artifact-free evaluation epochs for ERD/ERS analysis.

    The processing steps are similar to ``load_epochs``, but the electrode
    montage is added and the epochs are returned as an MNE ``Epochs`` object
    instead of a NumPy array. Epochs extend from -2.0 to 4.0 seconds relative
    to the motor-imagery cue, and common average reference is applied directly
    to the MNE object.

    This function only loads evaluation recordings. True labels are obtained
    from the corresponding MATLAB file, and trials marked with GDF event 1023
    are excluded.
    """
    file_path = Path(file_path)

    raw = load_raw_gdf(file_path)

    raw_eeg = pick_eeg_channels(raw)

    set_bci_2a_montage(raw_eeg)

    apply_bandpass_filter(raw_eeg)

    events, event_id = extract_events(raw)

    is_eval = is_eval_file(file_path)

    event_id_used = get_event_ids_for_session(
        event_id,
        is_eval,
    )

    if event_id_used is None:
        return None, None

    # Select only cue events belonging to clean trials.
    clean_cue_events, clean_trial_mask = (
        select_artifact_free_cue_events(
            events=events,
            event_id=event_id,
            event_id_used=event_id_used,
        )
    )

    epochs = create_epochs(
        raw_eeg,
        clean_cue_events,
        event_id_used,
        tmin=-2.0,
        tmax=4.0,
    )

    # Apply CAR
    epochs.set_eeg_reference(
        ref_channels="average",
        projection=False,
        verbose=False,
    )

    all_labels = load_true_labels(mat_path)

    if len(all_labels) != len(clean_trial_mask):
        raise ValueError(
            "The number of evaluation labels does not match "
            "the number of motor-imagery cue events: "
            f"{len(all_labels)} labels and "
            f"{len(clean_trial_mask)} cues."
        )

    # Remove labels corresponding to artifact-marked trials.
    clean_labels = all_labels[
        clean_trial_mask
    ]

    if len(epochs) != len(clean_labels):
        raise ValueError(
            "MNE dropped additional epochs after artifact removal: "
            f"{len(clean_labels)} clean labels but "
            f"{len(epochs)} retained epochs."
        )

    y = clean_labels

    return epochs, y