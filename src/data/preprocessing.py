import mne
import numpy as np
from mne import Epochs
from mne.io import BaseRaw

from src.utils.config import (
    EPOCH_TMAX,
    EPOCH_TMIN,
    H_FREQ,
    L_FREQ,
    NORMALIZATION_EPS,
    N_EEG_CHANNELS,
)


TRIAL_START_EVENT_CODE = "768"

EVALUATION_CUE_CODE = "783"

TRAINING_EVENT_CODES = {
    "left_hand": "769",
    "right_hand": "770",
    "feet": "771",
    "tongue": "772",
}

ARTIFACT_EVENT_CODE = "1023"

BCI_2A_CHANNEL_NAMES = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]


def select_artifact_free_cue_events(
    events: np.ndarray,
    event_id: dict[str, int],
    event_id_used: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO
    Select cue events belonging to trials not marked as artifacts.

    Parameters
    ----------
    events
        Complete MNE event array.

    event_id
        Mapping from original GDF annotation names, such as
        "768" and "1023", to MNE's internal event identifiers.

    event_id_used
        Cue events used to construct the epochs.

    Returns
    -------
    clean_cue_events
        Cue events belonging to artifact-free trials.

    clean_trial_mask
        Boolean mask aligned with all cue events. True indicates
        an artifact-free trial.
    """
    cue_event_ids = np.asarray(
        list(event_id_used.values()),
        dtype=int,
    )

    all_cue_events = events[
        np.isin(events[:, 2], cue_event_ids)
    ]

    if len(all_cue_events) == 0:
        raise ValueError("No motor-imagery cue events were found.")

    # Some recordings may contain no rejected trials.
    if ARTIFACT_EVENT_CODE not in event_id:
        clean_trial_mask = np.ones(
            len(all_cue_events),
            dtype=bool,
        )

        print("Artifact markers found: 0")

        return all_cue_events, clean_trial_mask

    if TRIAL_START_EVENT_CODE not in event_id:
        raise KeyError(
            f"Trial-start event {TRIAL_START_EVENT_CODE} "
            "was not found."
        )

    trial_start_id = event_id[TRIAL_START_EVENT_CODE]
    artifact_id = event_id[ARTIFACT_EVENT_CODE]

    trial_start_samples = events[
        events[:, 2] == trial_start_id,
        0,
    ]

    artifact_samples = events[
        events[:, 2] == artifact_id,
        0,
    ]

    if len(trial_start_samples) == 0:
        raise ValueError("No trial-start events were found.")

    # Associate every cue with the most recent trial-start event.
    cue_start_indices = (
        np.searchsorted(
            trial_start_samples,
            all_cue_events[:, 0],
            side="right",
        )
        - 1
    )

    if np.any(cue_start_indices < 0):
        raise ValueError(
            "At least one cue occurs before the first trial start."
        )

    cue_trial_starts = trial_start_samples[
        cue_start_indices
    ]

    # Associate every 1023 marker with its trial start.
    artifact_start_indices = (
        np.searchsorted(
            trial_start_samples,
            artifact_samples,
            side="right",
        )
        - 1
    )

    valid_artifact_indices = (
        artifact_start_indices >= 0
    )

    rejected_trial_starts = trial_start_samples[
        artifact_start_indices[valid_artifact_indices]
    ]

    clean_trial_mask = ~np.isin(
        cue_trial_starts,
        rejected_trial_starts,
    )

    clean_cue_events = all_cue_events[
        clean_trial_mask
    ]

    return clean_cue_events, clean_trial_mask


def set_bci_2a_montage(raw_eeg: BaseRaw) -> None:
    """
    TODO
    Assign the official BCI Competition IV 2a channel names and
    their standard 10-20 electrode positions.

    The GDF files do not necessarily contain channel names in a form
    that MNE can directly match to a standard montage.
    """
    if len(raw_eeg.ch_names) != len(BCI_2A_CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(BCI_2A_CHANNEL_NAMES)} EEG channels, "
            f"but found {len(raw_eeg.ch_names)}."
        )

    rename_mapping = {
        original_name: standard_name
        for original_name, standard_name in zip(
            raw_eeg.ch_names,
            BCI_2A_CHANNEL_NAMES,
        )
    }

    raw_eeg.rename_channels(rename_mapping)

    montage = mne.channels.make_standard_montage(
        "standard_1020"
    )

    raw_eeg.set_montage(
        montage,
        match_case=False,
        on_missing="raise",
    )


def pick_eeg_channels(raw: BaseRaw) -> BaseRaw:
    """
    Select the EEG channels and exclude the three EOG channels.

    In the BCI Competition IV 2a dataset, the first 22 channels are EEG
    channels and the final three channels are EOG channels.
    """
    return raw.copy().pick(raw.ch_names[:N_EEG_CHANNELS])


def apply_bandpass_filter(raw_eeg: BaseRaw) -> None:
    """
    Apply an 8–30 Hz zero-phase, fourth-order Butterworth band-pass filter.

    This frequency range retains the sensorimotor mu and beta rhythms,
    which are commonly modulated during motor imagery.
    """
    raw_eeg.filter(
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        method="iir",
        iir_params={
            "order": 4,
            "ftype": "butter",
        },
        phase="zero",
        verbose=False,
    )


def apply_car(X: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to the EEG epochs.

    CAR subtracts the average signal of all EEG channels from each channel.
    This helps reduce activity and noise shared across the electrodes and
    emphasizes the relative differences between brain regions.
    """
    return X - X.mean(axis=1, keepdims=True)


def extract_events(raw: BaseRaw) -> tuple[np.ndarray, dict[str, int]]:
    """
    Extract events and their internal MNE identifiers from annotations.

    Each event contains its sample position, previous value, and internal
    MNE event code.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    return events, event_id


def get_event_ids_for_session(
    event_id: dict[str, int],
    is_eval: bool,
) -> dict[str, int] | None:
    """
    Select the motor imagery event identifiers for a recording.

    Training recordings contain separate cue events for the four motor
    imagery classes. Evaluation recordings use the unknown cue event 783.
    """
    if is_eval:
        if EVALUATION_CUE_CODE not in event_id:
            print(
                f"No evaluation cue event "
                f"{EVALUATION_CUE_CODE} found."
            )
            return None

        return {
            "unknown_cue": event_id[EVALUATION_CUE_CODE]
        }

    missing_events = [
        event_code
        for event_code in TRAINING_EVENT_CODES.values()
        if event_code not in event_id
    ]

    if missing_events:
        print(
            "Missing training cue events: "
            f"{', '.join(missing_events)}"
        )
        return None

    return {
        class_name: event_id[event_code]
        for class_name, event_code 
        in TRAINING_EVENT_CODES.items()
    }


def create_epochs(
    raw_eeg: BaseRaw,
    events: np.ndarray,
    event_id_used: dict[str, int],
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
) -> Epochs:
    """
    Divide the continuous EEG recording into motor imagery epochs.

    Each epoch contains the signal from tmin to tmax seconds relative
    to the onset of its motor imagery cue.
    """
    return mne.Epochs(
        raw_eeg,
        events,
        event_id=event_id_used,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )


def get_epochs_data(epochs: Epochs) -> np.ndarray:
    """
    Return the epoched EEG signals as a NumPy array.

    The resulting shape is:
    (number of trials, number of channels, number of time samples).
    """
    return epochs.get_data()


def normalize_epochs(
    X_train: np.ndarray,
    X_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize the training and evaluation epochs channel-wise.

    The mean and standard deviation are calculated only from the training
    set and then applied to both the training and evaluation sets.
    """
    mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
    )
    std = (
        X_train.std(
            axis=(0, 2),
            keepdims=True,
        )
        + NORMALIZATION_EPS
    )

    X_train_normalized = (X_train - mean) / std
    X_eval_normalized = (X_eval - mean) / std

    return X_train_normalized, X_eval_normalized


def prepare_eegnet_input(X: np.ndarray) -> np.ndarray:
    """
    Add the final dimension required by the EEGNet implementation.

    The shape changes from (trials, channels, samples) to
    (trials, channels, samples, 1).
    """
    return X[..., np.newaxis]