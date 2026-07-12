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


EVALUATION_CUE_CODE = "783"

TRAINING_EVENT_CODES = {
    "left_hand": "769",
    "right_hand": "770",
    "feet": "771",
    "tongue": "772",
}


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
) -> Epochs:
    """
    Divide the continuous EEG recording into motor imagery epochs.

    Each epoch contains the signal from 0.5 to 4.0 seconds relative
    to the onset of its motor imagery cue.
    """
    return mne.Epochs(
        raw_eeg,
        events,
        event_id=event_id_used,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
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