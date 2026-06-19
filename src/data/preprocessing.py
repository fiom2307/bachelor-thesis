import mne
import numpy as np

def pick_eeg_channels(raw):
    # First 22 channels are EEG, last 3 are EOG
    raw_eeg = raw.copy().pick(raw.ch_names[:22])
    return raw_eeg

def apply_bandpass_filter(raw_eeg):
    # Apply a band-pass filter from 8 Hz to 30 Hz
    raw_eeg.filter(l_freq=8, h_freq=30, verbose=False)

def extract_events(raw):
    # events: [sample_position = 98241, previous_value = 0, event_code = 7]
    # event_id: {"769": 7, "770": 8, ...}
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    return events, event_id

def get_event_ids_for_session(event_id, is_eval):
    if is_eval:
        if "783" not in event_id:
            print(f"No cue event 783 found")
            return None

        return {
            "unknown_cue": event_id["783"]
        }

    if "769" not in event_id or "770" not in event_id:
        print(f"No cue events 769/770 found")
        return None

    return {
        "left_hand": event_id["769"],
        "right_hand": event_id["770"],
    }

def create_epochs(raw_eeg, events, event_id_used):
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

    return epochs

def get_epochs_data(epochs):
    # (n_trials = 144, 22, n_times = 1001 -> 4s = 1000)
    return epochs.get_data()

def normalize_epochs(X_train, X_eval):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_eval_norm = (X_eval - mean) / std

    return X_train_norm, X_eval_norm

def prepare_eegnet_input(X):
    return X[..., np.newaxis]