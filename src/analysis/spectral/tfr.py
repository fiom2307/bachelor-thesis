from collections.abc import Mapping
from dataclasses import dataclass

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

from src.analysis.spectral._utils import (
    TimeWindow,
    _compute_relative_power,
    _get_channel_data,
    _get_class_mask,
    _get_time_mask,
    _validate_epochs_and_labels,
)


@dataclass(frozen=True)
class ChannelTFRResult:
    """Store class-wise time-frequency results for one EEG channel.

    Attributes:
        power_by_class: Baseline-relative power for each class, with each
            array having shape ``frequencies × times``.
        n_trials_by_class: Number of trials included for each class.
        times: Time points after temporal decimation.
        freqs: Frequencies used for the Morlet transform.
        channel: Name of the analyzed EEG channel.
    """

    power_by_class: Mapping[str, np.ndarray]
    n_trials_by_class: Mapping[str, int]
    times: np.ndarray
    freqs: np.ndarray
    channel: str


def compute_channel_tfr(
    epochs: mne.Epochs,
    labels: np.ndarray,
    class_ids: Mapping[str, int],
    channel: str = "C3",
    frequencies: np.ndarray | None = None,
    baseline: TimeWindow = (-1.5, -0.5),
    decim: int = 2,
) -> ChannelTFRResult:
    """Compute class-wise baseline-normalized TFRs for one EEG channel.

    For each motor-imagery class, Morlet power is first averaged across all
    selected trials. Each frequency of that class-average TFR is then expressed
    as percentage change relative to its mean baseline power.

    Negative values indicate ERD and positive values indicate ERS.
    """
    labels = _validate_epochs_and_labels(epochs, labels)
    frequencies = (
        np.arange(8.0, 31.0, 1.0)
        if frequencies is None
        else np.asarray(frequencies)
    )

    channel_data = _get_channel_data(
        epochs,
        channel,
        keep_channel_axis=True,
    )
    sfreq = float(epochs.info["sfreq"])
    tfr_times = epochs.times[::decim]
    baseline_mask = _get_time_mask(tfr_times, baseline)

    power_by_class: dict[str, np.ndarray] = {}
    n_trials_by_class: dict[str, int] = {}

    for class_name, class_id in class_ids.items():
        class_mask, n_trials = _get_class_mask(
            labels,
            class_id,
            class_name,
        )

        power = tfr_array_morlet(
            data=channel_data[class_mask],
            sfreq=sfreq,
            freqs=frequencies,
            n_cycles=frequencies / 2.0,
            zero_mean=True,
            use_fft=True,
            decim=decim,
            output="power",
            verbose=False,
        )

        # power: trials × 1 channel × frequencies × times
        # mean_power: frequencies × times
        mean_power = power.mean(axis=(0, 1))
        power_by_class[class_name] = _compute_relative_power(
            mean_power,
            baseline_mask,
            context=f"class {class_name!r}",
        )
        n_trials_by_class[class_name] = n_trials

    return ChannelTFRResult(
        power_by_class=power_by_class,
        n_trials_by_class=n_trials_by_class,
        times=tfr_times,
        freqs=frequencies,
        channel=channel,
    )