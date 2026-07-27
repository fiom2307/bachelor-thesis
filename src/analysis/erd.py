from collections.abc import Mapping

from dataclasses import dataclass

import mne
import numpy as np
from mne.time_frequency import tfr_array_morlet


from src.data.labels import CLASS_LABELS


@dataclass(frozen=True)
class ERDResult:
    """
    ERD/ERS result for one subject, class, and frequency band.
    """

    time_course: np.ndarray
    topography: np.ndarray
    times: np.ndarray
    n_trials: int


ERDResults = dict[tuple[str, int], ERDResult]


def compute_class_erd(
    epochs: mne.Epochs,
    labels: np.ndarray,
    class_id: int,
    fmin: float,
    fmax: float,
    baseline: tuple[float, float] = (-1.5, -0.5),
    imagery_window: tuple[float, float] = (0.5, 3.5),
) -> ERDResult:
    """
    Calculate the average ERD/ERS for one motor-imagery class
    and one frequency band.
    TODO: check comments
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")

    if len(epochs) != len(labels):
        raise ValueError(
            "The number of epochs and labels must be equal."
        )

    class_mask = labels == class_id
    n_trials = int(class_mask.sum())

    if n_trials == 0:
        raise ValueError(
            f"No trials found for class {class_id}."
        )

    # Shape:
    # trials × channels × time
    class_data = epochs.get_data()[class_mask]

    frequencies = np.arange(
        int(np.ceil(fmin)),
        int(np.floor(fmax)) + 1,
        dtype=float,
    )

    if len(frequencies) == 0:
        raise ValueError("The frequency range is empty.")

    # Approximately constant temporal window across frequencies.
    n_cycles = frequencies / 2.0

    # Shape:
    # trials × channels × frequencies × time
    power = tfr_array_morlet(
        class_data,
        sfreq=float(epochs.info["sfreq"]),
        freqs=frequencies,
        n_cycles=n_cycles,
        output="power",
        zero_mean=True,
        use_fft=True,
        n_jobs=1,
        verbose=False,
    )

    times = epochs.times

    baseline_mask = (
        (times >= baseline[0])
        & (times <= baseline[1])
    )

    imagery_mask = (
        (times >= imagery_window[0])
        & (times <= imagery_window[1])
    )

    if not np.any(baseline_mask):
        raise ValueError(
            f"Baseline {baseline} is outside the epoch interval "
            f"{epochs.tmin, epochs.tmax}."
        )

    if not np.any(imagery_mask):
        raise ValueError(
            f"Imagery window {imagery_window} is outside the "
            f"epoch interval {epochs.tmin, epochs.tmax}."
        )

    # Baseline power is calculated separately for every:
    # trial × channel × frequency.
    baseline_power = power[
        ...,
        baseline_mask,
    ].mean(
        axis=-1,
        keepdims=True,
    )

    epsilon = np.finfo(np.float64).eps

    # Percentage change relative to baseline.
    erd_per_trial = (
        100.0
        * (power - baseline_power)
        / np.maximum(baseline_power, epsilon)
    )

    # Average across trials and frequencies.
    # Result: channels × time.
    mean_erd = erd_per_trial.mean(
        axis=(0, 2)
    )

    # Average across the selected motor-imagery interval.
    # Result: one value per channel.
    topography = mean_erd[
        :,
        imagery_mask,
    ].mean(
        axis=1
    )

    return ERDResult(
        time_course=mean_erd,
        topography=topography,
        times=times.copy(),
        n_trials=n_trials,
    )

def compute_all_erd_results(
    epochs: mne.Epochs,
    labels: np.ndarray,
    frequency_bands: Mapping[str, tuple[float, float]],
    baseline: tuple[float, float],
    imagery_window: tuple[float, float],
) -> ERDResults:
    """
    Calculate ERD/ERS results for all motor-imagery classes
    and frequency bands.
    """
    results: ERDResults = {}

    for band_name, (fmin, fmax) in frequency_bands.items():
        for class_id in CLASS_LABELS:
            results[(band_name, class_id)] = compute_class_erd(
                epochs=epochs,
                labels=labels,
                class_id=class_id,
                fmin=fmin,
                fmax=fmax,
                baseline=baseline,
                imagery_window=imagery_window,
            )

    return results