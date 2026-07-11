import mne
import numpy as np

def pick_eeg_channels(raw):
    # First 22 channels are EEG, last 3 are EOG
    raw_eeg = raw.copy().pick(raw.ch_names[:22])
    return raw_eeg

def apply_bandpass_filter(raw_eeg):
    # Apply a band-pass filter from 8 Hz to 30 Hz
    raw_eeg.filter(
        l_freq=8,
        h_freq=30,
        method="iir",
        iir_params=dict(order=4, ftype="butter"),
        phase="zero",
        verbose=False
    )

def apply_car(X):
    return X - X.mean(axis=1, keepdims=True)

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

    required_events = {
        "left_hand": "769",
        "right_hand": "770",
        "feet": "771",
        "tongue": "772",
    }

    missing_events = [
        event_code
        for event_code in required_events.values()
        if event_code not in event_id
    ]

    if missing_events:
        print(f"Missing training cue events: {', '.join(missing_events)}")
        return None

    return {
        class_name: event_id[event_code]
        for class_name, event_code in required_events.items()
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
        tmin=0.5, # was 0 before
        tmax=4, #was 4 before
        baseline=None,
        preload=True,
        verbose=False
    )

    return epochs

def get_epochs_data(epochs):
    # (n_trials = 144, 22, n_times = 1001 -> 4s = 1000) # ya no es asi
    return epochs.get_data()

def normalize_epochs(X_train, X_eval):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_eval_norm = (X_eval - mean) / std

    return X_train_norm, X_eval_norm

def prepare_eegnet_input(X):
    return X[..., np.newaxis]