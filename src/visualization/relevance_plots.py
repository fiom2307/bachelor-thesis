from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure

from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)


RelevanceMethod = Literal[
    "shap",
    "csp",
]

ChannelTimeRelevance = Mapping[
    int,
    np.ndarray,
]

VectorRelevance = Mapping[
    int,
    np.ndarray,
]

TrialCounts = Mapping[
    int,
    int,
]

ChannelRankings = Mapping[
    int,
    list[tuple[str, float]],
]


@dataclass(frozen=True)
class RelevancePlotConfig:
    """
    Plot labels for one relevance-analysis method.
    """

    channel_time_title: str
    temporal_title: str
    topographies_title: str
    channel_relevance_title: str
    ranking_method_name: str
    relevance_label: str
    time_label: str


PLOT_CONFIGS = {
    "shap": RelevancePlotConfig(
        channel_time_title=(
            "EEGNet channel–time SHAP relevance"
        ),
        temporal_title=(
            "EEGNet temporal SHAP relevance"
        ),
        topographies_title=(
            "EEGNet SHAP topographies"
        ),
        channel_relevance_title=(
            "Class-wise EEG channel SHAP relevance"
        ),
        ranking_method_name=(
            "SHAP relevance"
        ),
        relevance_label=(
            "Mean absolute SHAP value"
        ),
        time_label=(
            "Time relative to cue (s)"
        ),
    ),
    "csp": RelevancePlotConfig(
        channel_time_title=(
            "CSP+LDA channel–time occlusion relevance"
        ),
        temporal_title=(
            "CSP+LDA temporal occlusion relevance"
        ),
        topographies_title=(
            "CSP+LDA occlusion topographies"
        ),
        channel_relevance_title=(
            "Class-wise CSP+LDA channel relevance"
        ),
        ranking_method_name=(
            "CSP+LDA occlusion relevance"
        ),
        relevance_label=(
            "Mean absolute probability change"
        ),
        time_label=(
            "Occlusion-window center time (s)"
        ),
    ),
}


def plot_channel_time_relevance(
    class_relevance: ChannelTimeRelevance,
    times: np.ndarray,
    channel_names: list[str],
    method: RelevanceMethod,
    imagery_window: tuple[float, float] = (0.5, 4.0),
    trial_counts: TrialCounts | None = None,
) -> Figure:
    """
    Plot class-wise relevance across EEG channels and time.
    """
    config = _get_plot_config(
        method
    )

    class_ids = _get_class_ids(
        class_relevance
    )

    _validate_class_relevance(
        class_ids
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    maximum_relevance = max(
        np.max(
            class_relevance[
                class_id
            ]
        )
        for class_id in class_ids
    )

    maximum_relevance = max(
        float(maximum_relevance),
        np.finfo(float).eps,
    )

    image = None

    for axis, class_id in zip(
        axes,
        class_ids,
        strict=False,
    ):
        relevance = class_relevance[
            class_id
        ]

        if relevance.shape != (
            len(channel_names),
            len(times),
        ):
            raise ValueError(
                "Each channel-time relevance matrix must "
                "have shape (channels, times)."
            )

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

        axis.set_xlabel(
            config.time_label
        )

        axis.set_ylabel(
            "EEG channel"
        )

        axis.set_yticks(
            np.arange(
                len(channel_names)
            )
        )

        axis.set_yticklabels(
            channel_names,
            fontsize=8,
        )

        _mark_imagery_window(
            axis,
            imagery_window,
        )

    for axis in axes[
        len(class_ids):
    ]:
        axis.set_visible(False)

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.85,
        )

        colorbar.set_label(
            config.relevance_label
        )

    figure.suptitle(
        config.channel_time_title,
        fontsize=15,
    )

    return figure


def plot_temporal_relevance(
    temporal_relevance: VectorRelevance,
    times: np.ndarray,
    method: RelevanceMethod,
    imagery_window: tuple[float, float] = (0.5, 4.0),
    trial_counts: TrialCounts | None = None,
) -> Figure:
    """
    Plot temporal relevance averaged across EEG channels.
    """
    config = _get_plot_config(
        method
    )

    class_ids = _get_class_ids(
        temporal_relevance
    )

    _validate_class_relevance(
        class_ids
    )

    figure, axis = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True,
    )

    for class_id in class_ids:
        relevance = temporal_relevance[
            class_id
        ]

        if relevance.shape != times.shape:
            raise ValueError(
                "Each temporal relevance vector must "
                "have the same shape as times."
            )

        axis.plot(
            times,
            relevance,
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
        config.temporal_title
    )

    axis.set_xlabel(
        config.time_label
    )

    axis.set_ylabel(
        config.relevance_label
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


def plot_topographies(
    topographic_relevance: VectorRelevance,
    info: mne.Info,
    method: RelevanceMethod,
    imagery_window: tuple[float, float] = (0.5, 4.0),
    trial_counts: TrialCounts | None = None,
    show_channel_names: bool = False,
) -> Figure:
    """
    Plot class-wise scalp topographies of model relevance.
    """
    config = _get_plot_config(
        method
    )

    class_ids = _get_class_ids(
        topographic_relevance
    )

    _validate_class_relevance(
        class_ids
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    maximum_relevance = max(
        np.max(
            topographic_relevance[
                class_id
            ]
        )
        for class_id in class_ids
    )

    maximum_relevance = max(
        float(maximum_relevance),
        np.finfo(float).eps,
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
        relevance = topographic_relevance[
            class_id
        ]

        if len(relevance) != len(
            info.ch_names
        ):
            raise ValueError(
                "The number of topographic relevance "
                "values must match the number of channels."
            )

        image, _ = mne.viz.plot_topomap(
            data=relevance,
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

    for axis in axes[
        len(class_ids):
    ]:
        axis.set_visible(False)

    if image is not None:
        colorbar = figure.colorbar(
            image,
            ax=axes.tolist(),
            shrink=0.8,
        )

        colorbar.set_label(
            config.relevance_label
        )

    window_start, window_end = (
        imagery_window
    )

    figure.suptitle(
        f"{config.topographies_title} "
        f"({window_start:.1f}–{window_end:.1f} s)",
        fontsize=15,
    )

    return figure


def plot_channel_relevance(
    channel_relevance: VectorRelevance,
    channel_names: list[str],
    method: RelevanceMethod,
    trial_counts: TrialCounts | None = None,
) -> Figure:
    """
    Plot a class-by-channel relevance heatmap.
    """
    config = _get_plot_config(
        method
    )

    class_ids = _get_class_ids(
        channel_relevance
    )

    _validate_class_relevance(
        class_ids
    )

    relevance_matrix = np.stack([
        channel_relevance[
            class_id
        ]
        for class_id in class_ids
    ])

    if relevance_matrix.shape[1] != len(
        channel_names
    ):
        raise ValueError(
            "The number of relevance values must "
            "match the number of channel names."
        )

    figure, axis = plt.subplots(
        figsize=(15, 5),
        constrained_layout=True,
    )

    maximum_relevance = max(
        float(
            np.max(
                relevance_matrix
            )
        ),
        np.finfo(float).eps,
    )

    image = axis.imshow(
        relevance_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=maximum_relevance,
    )

    axis.set_xticks(
        np.arange(
            len(channel_names)
        )
    )

    axis.set_xticklabels(
        channel_names,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(
            len(class_ids)
        )
    )

    axis.set_yticklabels([
        _get_class_title(
            class_id,
            trial_counts,
        )
        for class_id in class_ids
    ])

    axis.set_title(
        config.channel_relevance_title
    )

    axis.set_xlabel(
        "EEG channel"
    )

    axis.set_ylabel(
        "Motor-imagery class"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.9,
    )

    colorbar.set_label(
        config.relevance_label
    )

    return figure


def plot_channel_rankings(
    channel_rankings: ChannelRankings,
    method: RelevanceMethod,
    top_n: int = 10,
    trial_counts: TrialCounts | None = None,
) -> Figure:
    """
    Plot the highest-ranked EEG channels for each class.
    """
    config = _get_plot_config(
        method
    )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    class_ids = [
        class_id
        for class_id in CLASS_LABELS
        if class_id in channel_rankings
    ]

    _validate_class_relevance(
        class_ids
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for axis, class_id in zip(
        axes,
        class_ids,
        strict=False,
    ):
        ranking = channel_rankings[
            class_id
        ][:top_n]

        channel_names = [
            channel_name
            for channel_name, _
            in ranking
        ]

        relevance_values = [
            relevance
            for _, relevance
            in ranking
        ]

        axis.barh(
            channel_names,
            relevance_values,
        )

        axis.invert_yaxis()

        axis.set_title(
            _get_class_title(
                class_id,
                trial_counts,
            )
        )

        axis.set_xlabel(
            config.relevance_label
        )

        axis.set_ylabel(
            "EEG channel"
        )

        axis.grid(
            axis="x",
            alpha=0.25,
        )

    for axis in axes[
        len(class_ids):
    ]:
        axis.set_visible(False)

    figure.suptitle(
        f"Top {top_n} EEG channels by "
        f"{config.ranking_method_name}",
        fontsize=15,
    )

    return figure


def _get_plot_config(
    method: RelevanceMethod,
) -> RelevancePlotConfig:
    """
    Return the plot configuration for one analysis method.
    """
    try:
        return PLOT_CONFIGS[
            method
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported relevance method: {method}"
        ) from error


def _get_class_ids(
    relevance: Mapping[
        int,
        object,
    ],
) -> list[int]:
    """
    Return available classes in the predefined class order.
    """
    return [
        class_id
        for class_id in CLASS_LABELS
        if class_id in relevance
    ]


def _get_class_title(
    class_id: int,
    trial_counts: TrialCounts | None,
) -> str:
    """
    Return the readable class name and optional trial count.
    """
    class_index = list(
        CLASS_LABELS
    ).index(class_id)

    class_name = CLASS_NAMES[
        class_index
    ]

    if trial_counts is None:
        return class_name

    trial_count = trial_counts.get(
        class_id,
        0,
    )

    return (
        f"{class_name} "
        f"(n={trial_count})"
    )


def _validate_class_relevance(
    class_ids: list[int],
) -> None:
    """
    Ensure that relevance data contains at least one class.
    """
    if not class_ids:
        raise ValueError(
            "No class relevance data is available."
        )


def _mark_imagery_window(
    axis: plt.Axes,
    imagery_window: tuple[float, float],
) -> None:
    """
    Mark the motor-imagery interval.
    """
    window_start, window_end = (
        imagery_window
    )

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