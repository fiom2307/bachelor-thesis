import mne
import numpy as np


TimeWindow = tuple[float, float]


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