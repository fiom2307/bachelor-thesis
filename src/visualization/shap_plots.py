from collections.abc import Mapping

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure

from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)


ChannelTimeRelevance = Mapping[int, np.ndarray]
VectorRelevance = Mapping[int, np.ndarray]
TrialCounts = Mapping[int, int]


def plot_shap_channel_time(
    class_relevance: ChannelTimeRelevance,
    times: np.ndarray,
    channel_names: list[str],
    imagery_window: tuple[float, float] = (0.5, 3.5),
    trial_counts: TrialCounts | None = None,
) -> Figure:
    """
    Plot class-wise SHAP relevance across EEG channels and time.
    """
    class_ids = _get_class_ids(class_relevance)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    maximum_relevance = max(
        np.max(class_relevance[class_id])
        for class_id in class_ids
    )

    image = None

    for axis, class_id in zip(
        axes,
        class_ids,
        strict=False,
    ):
        relevance = class_relevance[class_id]

        image = axis.imshow(
            relevance,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            extent=(
                times[0],
                times[-1],
                len(channel_names) - 0.5,
                -0.5,
            ),
            cmap="viridis",
            vmin=0.0,
            vmax=maximum_relevance,
        )

        axis.set_title(
            _get_class_title(
                class_id,
                trial_counts,
            )
        )

        axis.set_xlabel("Time relative to cue (s)")
        axis.set_ylabel("EEG channel")

        axis.set_yticks(
            np.arange(len(channel_names))
        )

        axis.set_yticklabels(
            channel_names,
            fontsize=8,
        )

        _mark_imagery_window(
            axis,
            imagery_window,
        )

    for axis in axes[len(class_ids):]:
        axis.set_visible(False)

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.85,
        )

        colorbar.set_label(
            "Mean absolute SHAP value"
        )

    figure.suptitle(
        "EEGNet channel–time SHAP relevance",
        fontsize=15,
    )

    return figure


def plot_shap_temporal_relevance(
    temporal_relevance: VectorRelevance,
    times: np.ndarray,
    imagery_window: tuple[float, float] = (0.5, 3.5),
    trial_counts: TrialCounts | None = None,
) -> Figure:
    """
    Plot temporal SHAP relevance averaged across EEG channels.
    """
    class_ids = _get_class_ids(
        temporal_relevance
    )

    figure, axis = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True,
    )

    for class_id in class_ids:
        axis.plot(
            times,
            temporal_relevance[class_id],
            linewidth=2,
            label=_get_class_title(
                class_id,
                trial_counts,
            ),
        )

    _mark_imagery_window(
        axis,
        imagery_window,
    )

    axis.set_title(
        "Temporal SHAP relevance"
    )

    axis.set_xlabel(
        "Time relative to cue (s)"
    )

    axis.set_ylabel(
        "Mean absolute SHAP value"
    )

    axis.set_xlim(
        times[0],
        times[-1],
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend(
        title="Motor-imagery class",
    )

    return figure


def plot_shap_topographies(
    topographic_relevance: VectorRelevance,
    info: mne.Info,
    imagery_window: tuple[float, float] = (0.5, 3.5),
    trial_counts: TrialCounts | None = None,
    show_channel_names: bool = False,
) -> Figure:
    """
    Plot class-wise scalp topographies of SHAP relevance.
    """
    class_ids = _get_class_ids(
        topographic_relevance
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    maximum_relevance = max(
        np.max(topographic_relevance[class_id])
        for class_id in class_ids
    )

    channel_names = (
        info.ch_names
        if show_channel_names
        else None
    )

    image = None

    for axis, class_id in zip(
        axes,
        class_ids,
        strict=False,
    ):
        image, _ = mne.viz.plot_topomap(
            data=topographic_relevance[class_id],
            pos=info,
            axes=axis,
            show=False,
            sensors=True,
            names=channel_names,
            contours=6,
            cmap="viridis",
            vlim=(
                0.0,
                maximum_relevance,
            ),
        )

        axis.set_title(
            _get_class_title(
                class_id,
                trial_counts,
            )
        )

    for axis in axes[len(class_ids):]:
        axis.set_visible(False)

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.8,
        )

        colorbar.set_label(
            "Mean absolute SHAP value"
        )

    window_start, window_end = imagery_window

    figure.suptitle(
        "EEGNet SHAP topographies "
        f"({window_start:.1f}–{window_end:.1f} s)",
        fontsize=15,
    )

    return figure


def _get_class_ids(
    relevance: Mapping[int, np.ndarray],
) -> list[int]:
    """Return available class labels in the predefined class order."""
    return [
        class_id
        for class_id in CLASS_LABELS
        if class_id in relevance
    ]


def _get_class_title(
    class_id: int,
    trial_counts: TrialCounts | None,
) -> str:
    """Return the readable class name and optional trial count."""
    class_index = list(
        CLASS_LABELS
    ).index(class_id)

    class_name = CLASS_NAMES[
        class_index
    ]

    if trial_counts is None:
        return class_name

    return (
        f"{class_name} "
        f"(n={trial_counts[class_id]})"
    )


def _mark_imagery_window(
    axis: plt.Axes,
    imagery_window: tuple[float, float],
) -> None:
    """Mark the motor-imagery interval."""
    window_start, window_end = imagery_window

    axis.axvline(
        window_start,
        linestyle="--",
        linewidth=1,
        alpha=0.8,
    )

    axis.axvline(
        window_end,
        linestyle="--",
        linewidth=1,
        alpha=0.8,
    )