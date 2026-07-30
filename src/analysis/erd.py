from collections.abc import Mapping
from dataclasses import dataclass

import mne
import numpy as np
from mne.time_frequency import psd_array_welch, tfr_array_morlet

from src.data.labels import CLASS_LABELS


TimeWindow = tuple[float, float]
ERDResults = dict[tuple[str, int], "ERDResult"]


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


# ---------------------------------------------------------------------------
# Validation and shared calculations
# ---------------------------------------------------------------------------


def _validate_epochs_and_labels(
    epochs: mne.Epochs,
    labels: np.ndarray,
) -> np.ndarray:
    """Validate the labels and return them as a one-dimensional array.

    The number of labels must match the number of epochs because every epoch
    is expected to have exactly one motor-imagery class label.
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array.")

    if len(epochs) != len(labels):
        raise ValueError(
            "The number of epochs and labels must be equal."
        )

    return labels


def _get_class_mask(
    labels: np.ndarray,
    class_id: int,
    class_name: str | None = None,
) -> tuple[np.ndarray, int]:
    """Return the trial mask and trial count for one class.

    Args:
        labels: One-dimensional class-label array.
        class_id: Numeric identifier of the requested class.
        class_name: Optional readable name used only in error messages.
    """
    class_mask = labels == class_id
    n_trials = int(class_mask.sum())

    if n_trials == 0:
        class_description = (
            f"class {class_name!r}"
            if class_name is not None
            else f"class {class_id}"
        )
        raise ValueError(
            f"No trials were found for {class_description}."
        )

    return class_mask, n_trials


def _get_time_mask(
    times: np.ndarray,
    window: TimeWindow,
) -> np.ndarray:
    """Return a Boolean mask selecting an inclusive time window.

    Raises:
        ValueError: If the requested interval does not overlap the available
            epoch time points.
    """
    start, end = window
    mask = (times >= start) & (times <= end)

    if not np.any(mask):
        raise ValueError(
            f"Window {window} does not overlap with epoch times "
            f"{times[0]:.3f} to {times[-1]:.3f} s."
        )

    return mask


def _get_channel_data(
    epochs: mne.Epochs,
    channel: str,
    *,
    keep_channel_axis: bool,
) -> np.ndarray:
    """Extract all trials from one EEG channel.

    Args:
        epochs: Preprocessed EEG epochs.
        channel: Name of the channel to extract.
        keep_channel_axis: Keep a singleton channel dimension when ``True``.
            This is required by ``tfr_array_morlet``.

    Returns:
        Data with shape ``trials × 1 × times`` when ``keep_channel_axis`` is
        true, otherwise ``trials × times``.
    """
    if channel not in epochs.ch_names:
        raise ValueError(f"Channel {channel!r} was not found.")

    channel_index = epochs.ch_names.index(channel)
    data = epochs.get_data()

    if keep_channel_axis:
        return data[:, channel_index : channel_index + 1, :]

    return data[:, channel_index, :]


def _compute_relative_power(
    power: np.ndarray,
    baseline_mask: np.ndarray,
    *,
    context: str = "",
) -> np.ndarray:
    """Express power as percentage change relative to its baseline.

    The baseline is calculated along the final time axis. Negative values
    indicate event-related desynchronization (ERD), while positive values
    indicate event-related synchronization (ERS).
    """
    baseline_power = power[..., baseline_mask].mean(
        axis=-1,
        keepdims=True,
    )

    error_suffix = f" for {context}" if context else ""

    if not np.all(np.isfinite(baseline_power)):
        raise ValueError(
            "Baseline power contains non-finite values"
            f"{error_suffix}."
        )

    if np.any(baseline_power <= 0.0):
        raise ValueError(
            "Baseline power must be strictly positive"
            f"{error_suffix}."
        )

    return 100.0 * (power - baseline_power) / baseline_power


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


# ---------------------------------------------------------------------------
# ERD/ERS analysis across channels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Single-channel time-frequency analysis
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Single-channel spectral analysis
# ---------------------------------------------------------------------------


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