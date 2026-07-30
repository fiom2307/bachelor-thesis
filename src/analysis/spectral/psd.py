from collections.abc import Mapping
from dataclasses import dataclass

import mne
import numpy as np
from mne.time_frequency import psd_array_welch

from src.analysis.spectral._utils import (
    TimeWindow,
    _get_channel_data,
    _get_class_mask,
    _get_time_mask,
    _validate_epochs_and_labels,
)


@dataclass(frozen=True)
class ChannelPSDResult:
    """Store class-wise and baseline PSD statistics for one EEG channel.

    Attributes:
        freqs: Frequencies returned by Welch's method.
        mean_by_condition: Mean PSD for every motor-imagery class and the
            baseline condition.
        sem_by_condition: Standard error of the PSD for every condition.
        n_trials_by_condition: Number of trials used for every condition.
        channel: Name of the analyzed EEG channel.
    """

    freqs: np.ndarray
    mean_by_condition: Mapping[str, np.ndarray]
    sem_by_condition: Mapping[str, np.ndarray]
    n_trials_by_condition: Mapping[str, int]
    channel: str


def _compute_welch_psd(
    data: np.ndarray,
    *,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_fft: int,
    n_per_seg: int,
    n_overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD values and convert them from V²/Hz to µV²/Hz."""
    psd, freqs = psd_array_welch(
        x=data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        average="mean",
        window="hamming",
        verbose=False,
    )

    return psd * 1e12, freqs


def _mean_and_sem(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the trial-wise mean and standard error of the mean.

    A zero-valued SEM is returned when fewer than two trials are available,
    because a sample standard deviation cannot be estimated in that case.
    """
    mean = values.mean(axis=0)

    if len(values) < 2:
        sem = np.zeros_like(mean)
    else:
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(values))

    return mean, sem


def compute_channel_psd(
    epochs: mne.Epochs,
    labels: np.ndarray,
    class_ids: Mapping[str, int],
    channel: str = "C3",
    baseline: TimeWindow = (-1.5, -0.5),
    imagery_window: TimeWindow = (0.5, 3.5),
    fmin: float = 8.0,
    fmax: float = 30.0,
) -> ChannelPSDResult:
    """Compute Welch PSD statistics for one channel and all MI classes.

    Each motor-imagery condition uses the samples inside ``imagery_window``
    from trials of that class. The baseline condition uses the pre-cue segment
    from every retained trial. PSD values are returned in µV²/Hz together with
    their trial-wise mean and standard error.
    """
    labels = _validate_epochs_and_labels(epochs, labels)
    channel_data = _get_channel_data(
        epochs,
        channel,
        keep_channel_axis=False,
    )

    baseline_mask = _get_time_mask(epochs.times, baseline)
    imagery_mask = _get_time_mask(epochs.times, imagery_window)
    baseline_data = channel_data[:, baseline_mask]

    imagery_data_by_class: dict[str, np.ndarray] = {}
    n_trials_by_condition: dict[str, int] = {}

    for class_name, class_id in class_ids.items():
        class_mask, n_trials = _get_class_mask(
            labels,
            class_id,
            class_name,
        )
        imagery_data_by_class[class_name] = channel_data[
            class_mask
        ][:, imagery_mask]
        n_trials_by_condition[class_name] = n_trials

    n_trials_by_condition["Baseline"] = len(baseline_data)

    sfreq = float(epochs.info["sfreq"])
    imagery_length = next(iter(imagery_data_by_class.values())).shape[-1]
    n_per_seg = min(
        int(round(sfreq)),
        imagery_length,
        baseline_data.shape[-1],
    )
    n_fft = int(2 ** np.ceil(np.log2(2 * n_per_seg)))
    n_overlap = n_per_seg // 2

    psd_parameters = {
        "sfreq": sfreq,
        "fmin": fmin,
        "fmax": fmax,
        "n_fft": n_fft,
        "n_per_seg": n_per_seg,
        "n_overlap": n_overlap,
    }

    mean_by_condition: dict[str, np.ndarray] = {}
    sem_by_condition: dict[str, np.ndarray] = {}
    freqs: np.ndarray | None = None

    for class_name, class_data in imagery_data_by_class.items():
        class_psd, class_freqs = _compute_welch_psd(
            class_data,
            **psd_parameters,
        )
        mean_by_condition[class_name], sem_by_condition[class_name] = (
            _mean_and_sem(class_psd)
        )

        if freqs is None:
            freqs = class_freqs

    baseline_psd, baseline_freqs = _compute_welch_psd(
        baseline_data,
        **psd_parameters,
    )
    mean_by_condition["Baseline"], sem_by_condition["Baseline"] = (
        _mean_and_sem(baseline_psd)
    )

    if freqs is None:
        freqs = baseline_freqs

    return ChannelPSDResult(
        freqs=freqs,
        mean_by_condition=mean_by_condition,
        sem_by_condition=sem_by_condition,
        n_trials_by_condition=n_trials_by_condition,
        channel=channel,
    )