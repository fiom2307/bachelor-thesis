from collections.abc import Mapping
from dataclasses import dataclass

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet

from src.analysis.spectral._utils import (
    TimeWindow,
    _compute_relative_power,
    _get_class_mask,
    _get_time_mask,
    _validate_epochs_and_labels,
)
from src.data.labels import CLASS_LABELS


@dataclass(frozen=True)
class ERDResult:
    """Store the ERD/ERS outputs for one class and frequency band.

    Attributes:
        time_course: Baseline-relative ERD/ERS values with shape
            ``channels × times``.
        topography: Mean ERD/ERS value per channel during the imagery window.
        times: Time points corresponding to ``time_course``.
        n_trials: Number of artifact-free trials used in the calculation.
    """

    time_course: np.ndarray
    topography: np.ndarray
    times: np.ndarray
    n_trials: int


ERDResults = dict[tuple[str, int], "ERDResult"]


def compute_class_erd(
    epochs: mne.Epochs,
    labels: np.ndarray,
    class_id: int,
    fmin: float,
    fmax: float,
    baseline: TimeWindow = (-1.5, -0.5),
    imagery_window: TimeWindow = (0.5, 3.5),
) -> ERDResult:
    """Compute class-average ERD/ERS for one frequency band.

    Morlet power is computed for every artifact-free trial and EEG channel.
    Power is then averaged across the frequencies in the selected band and
    across all trials belonging to ``class_id``. The resulting channel-wise
    time course is expressed as percentage change relative to the baseline.

    The returned topography is the mean ERD/ERS value of each channel during
    ``imagery_window``.
    """
    labels = _validate_epochs_and_labels(epochs, labels)
    class_mask, n_trials = _get_class_mask(labels, class_id)

    frequencies = np.arange(
        int(np.ceil(fmin)),
        int(np.floor(fmax)) + 1,
        dtype=float,
    )
    if len(frequencies) == 0:
        raise ValueError("The frequency range is empty.")

    power = tfr_array_morlet(
        epochs.get_data()[class_mask],
        sfreq=float(epochs.info["sfreq"]),
        freqs=frequencies,
        n_cycles=frequencies / 2.0,
        output="power",
        zero_mean=True,
        use_fft=True,
        n_jobs=1,
        verbose=False,
    )

    times = epochs.times
    baseline_mask = _get_time_mask(times, baseline)
    imagery_mask = _get_time_mask(times, imagery_window)

    # power: trials × channels × frequencies × times
    # mean_band_power: channels × times
    mean_band_power = power.mean(axis=2).mean(axis=0)
    erd_ers_time_course = _compute_relative_power(
        mean_band_power,
        baseline_mask,
    )
    topography = erd_ers_time_course[:, imagery_mask].mean(axis=1)

    return ERDResult(
        time_course=erd_ers_time_course,
        topography=topography,
        times=times.copy(),
        n_trials=n_trials,
    )


def compute_all_erd_results(
    epochs: mne.Epochs,
    labels: np.ndarray,
    frequency_bands: Mapping[str, TimeWindow],
    baseline: TimeWindow,
    imagery_window: TimeWindow,
) -> ERDResults:
    """Compute ERD/ERS for every configured frequency band and MI class.

    Results are indexed by ``(band_name, class_id)`` so that the plotting code
    can retrieve each class-band combination directly.
    """
    return {
        (band_name, class_id): compute_class_erd(
            epochs=epochs,
            labels=labels,
            class_id=class_id,
            fmin=fmin,
            fmax=fmax,
            baseline=baseline,
            imagery_window=imagery_window,
        )
        for band_name, (fmin, fmax) in frequency_bands.items()
        for class_id in CLASS_LABELS
    }