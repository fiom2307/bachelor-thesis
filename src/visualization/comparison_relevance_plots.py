from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)


RelevanceMethod = Literal[
    "shap",
    "csp",
]

VectorRelevance = Mapping[
    int,
    np.ndarray,
]

NormalizedChannelFrequency = Mapping[
    int,
    Mapping[str, float],
]


@dataclass(frozen=True)
class ComparisonPlotConfig:
    """
    Labels for one group-level relevance method.
    """

    method_name: str
    time_label: str


PLOT_CONFIGS = {
    "shap": ComparisonPlotConfig(
        method_name="EEGNet SHAP",
        time_label="Time relative to cue (s)",
    ),
    "csp": ComparisonPlotConfig(
        method_name="CSP+LDA occlusion",
        time_label="Occlusion-window center time (s)",
    ),
}


def plot_group_mean_channel_relevance(
    mean_channel_relevance: VectorRelevance,
    channel_names: list[str],
    method: RelevanceMethod,
) -> Figure:
    """
    Plot mean normalized channel relevance across subjects.
    """
    config = _get_plot_config(
        method
    )

    class_ids, relevance_matrix = (
        _create_class_channel_matrix(
            relevance=mean_channel_relevance,
            channel_names=channel_names,
        )
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
        _get_class_name(
            class_id
        )
        for class_id in class_ids
    ])

    axis.set_title(
        f"{config.method_name}: "
        "mean channel relevance across subjects"
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
        "Mean normalized relevance"
    )

    return figure


def plot_group_channel_variability(
    channel_standard_deviation: VectorRelevance,
    channel_names: list[str],
    method: RelevanceMethod,
) -> Figure:
    """
    Plot between-subject variability in channel relevance.
    """
    config = _get_plot_config(
        method
    )

    class_ids, variability_matrix = (
        _create_class_channel_matrix(
            relevance=channel_standard_deviation,
            channel_names=channel_names,
        )
    )

    figure, axis = plt.subplots(
        figsize=(15, 5),
        constrained_layout=True,
    )

    maximum_variability = max(
        float(
            np.max(
                variability_matrix
            )
        ),
        np.finfo(float).eps,
    )

    image = axis.imshow(
        variability_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=maximum_variability,
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
        _get_class_name(
            class_id
        )
        for class_id in class_ids
    ])

    axis.set_title(
        f"{config.method_name}: "
        "between-subject channel variability"
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
        "Standard deviation of normalized relevance"
    )

    return figure


def plot_group_mean_temporal_relevance(
    mean_temporal_relevance: VectorRelevance,
    temporal_standard_deviation: VectorRelevance,
    times: np.ndarray,
    method: RelevanceMethod,
    imagery_window: tuple[float, float] = (0.5, 4.0),
) -> Figure:
    """
    Plot mean temporal relevance and variability across subjects.
    """
    config = _get_plot_config(
        method
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    _validate_times(
        times
    )

    class_ids = _get_shared_class_ids(
        mean_temporal_relevance,
        temporal_standard_deviation,
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 9),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    maximum_upper_bound = 0.0

    prepared_values = {}

    for class_id in class_ids:
        mean_values = np.asarray(
            mean_temporal_relevance[
                class_id
            ],
            dtype=np.float64,
        )

        standard_deviation = np.asarray(
            temporal_standard_deviation[
                class_id
            ],
            dtype=np.float64,
        )

        if mean_values.shape != times.shape:
            raise ValueError(
                f"Class {class_id}: mean temporal relevance "
                "must have the same shape as times."
            )

        if standard_deviation.shape != times.shape:
            raise ValueError(
                f"Class {class_id}: temporal standard "
                "deviation must have the same shape as times."
            )

        if not (
            np.all(
                np.isfinite(
                    mean_values
                )
            )
            and np.all(
                np.isfinite(
                    standard_deviation
                )
            )
        ):
            raise ValueError(
                f"Class {class_id}: temporal relevance "
                "contains non-finite values."
            )

        lower_bound = np.maximum(
            mean_values
            - standard_deviation,
            0.0,
        )

        upper_bound = (
            mean_values
            + standard_deviation
        )

        maximum_upper_bound = max(
            maximum_upper_bound,
            float(
                np.max(
                    upper_bound
                )
            ),
        )

        prepared_values[class_id] = (
            mean_values,
            lower_bound,
            upper_bound,
        )

    maximum_upper_bound = max(
        maximum_upper_bound,
        np.finfo(float).eps,
    )

    for axis, class_id in zip(
        axes,
        class_ids,
        strict=False,
    ):
        (
            mean_values,
            lower_bound,
            upper_bound,
        ) = prepared_values[
            class_id
        ]

        axis.plot(
            times,
            mean_values,
            linewidth=2,
            label="Mean",
        )

        axis.fill_between(
            times,
            lower_bound,
            upper_bound,
            alpha=0.25,
            label="±1 standard deviation",
        )

        _mark_imagery_window(
            axis,
            imagery_window,
        )

        axis.set_title(
            _get_class_name(
                class_id
            )
        )

        axis.set_xlabel(
            config.time_label
        )

        axis.set_ylabel(
            "Normalized relevance"
        )

        axis.set_xlim(
            times[0],
            times[-1],
        )

        axis.set_ylim(
            0.0,
            maximum_upper_bound * 1.05,
        )

        axis.grid(
            alpha=0.25,
        )

        axis.legend()

    for axis in axes[
        len(class_ids):
    ]:
        axis.set_visible(False)

    figure.suptitle(
        f"{config.method_name}: "
        "mean temporal relevance across subjects",
        fontsize=15,
    )

    return figure


def plot_top5_channel_frequency(
    normalized_frequency: NormalizedChannelFrequency,
    method: RelevanceMethod,
    display_n: int = 10,
) -> Figure:
    """
    Plot how frequently channels appear in the subject-wise top five.
    """
    config = _get_plot_config(
        method
    )

    if display_n <= 0:
        raise ValueError(
            "display_n must be greater than zero."
        )

    class_ids = [
        class_id
        for class_id in CLASS_LABELS
        if class_id in normalized_frequency
    ]

    if not class_ids:
        raise ValueError(
            "No channel-frequency results are available."
        )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for axis, class_id in zip(
        axes,
        class_ids,
        strict=False,
    ):
        class_frequency = normalized_frequency[
            class_id
        ]

        ranking = sorted(
            (
                (
                    channel_name,
                    float(frequency),
                )
                for channel_name, frequency
                in class_frequency.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        ranking = ranking[
            :display_n
        ]

        channel_names = [
            channel_name
            for channel_name, _
            in ranking
        ]

        frequencies = np.asarray(
            [
                frequency
                for _, frequency
                in ranking
            ],
            dtype=np.float64,
        )

        if np.any(
            (
                frequencies < 0.0
            )
            | (
                frequencies > 1.0
            )
        ):
            raise ValueError(
                "Normalized channel frequencies must "
                "be between zero and one."
            )

        axis.barh(
            channel_names,
            frequencies,
        )

        axis.invert_yaxis()

        axis.set_title(
            _get_class_name(
                class_id
            )
        )

        axis.set_xlabel(
            "Subjects with channel in top five"
        )

        axis.set_ylabel(
            "EEG channel"
        )

        axis.set_xlim(
            0.0,
            1.0,
        )

        axis.xaxis.set_major_formatter(
            PercentFormatter(
                xmax=1.0
            )
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
        f"{config.method_name}: "
        "frequency of top-five channels across subjects",
        fontsize=15,
    )

    return figure


def _create_class_channel_matrix(
    relevance: VectorRelevance,
    channel_names: list[str],
) -> tuple[
    list[int],
    np.ndarray,
]:
    """
    Create a class-by-channel relevance matrix.
    """
    _validate_channel_names(
        channel_names
    )

    class_ids = [
        class_id
        for class_id in CLASS_LABELS
        if class_id in relevance
    ]

    if not class_ids:
        raise ValueError(
            "No channel relevance data is available."
        )

    rows = []

    for class_id in class_ids:
        class_values = np.asarray(
            relevance[
                class_id
            ],
            dtype=np.float64,
        )

        if class_values.ndim != 1:
            raise ValueError(
                f"Class {class_id}: channel relevance "
                "must be one-dimensional."
            )

        if len(class_values) != len(
            channel_names
        ):
            raise ValueError(
                f"Class {class_id}: the number of "
                "relevance values must match the "
                "number of channel names."
            )

        if not np.all(
            np.isfinite(
                class_values
            )
        ):
            raise ValueError(
                f"Class {class_id}: channel relevance "
                "contains non-finite values."
            )

        rows.append(
            class_values
        )

    return (
        class_ids,
        np.stack(
            rows,
            axis=0,
        ),
    )


def _get_shared_class_ids(
    first_relevance: VectorRelevance,
    second_relevance: VectorRelevance,
) -> list[int]:
    """
    Return classes available in both relevance mappings.
    """
    class_ids = [
        class_id
        for class_id in CLASS_LABELS
        if (
            class_id in first_relevance
            and class_id in second_relevance
        )
    ]

    if not class_ids:
        raise ValueError(
            "No shared class relevance data is available."
        )

    return class_ids


def _get_plot_config(
    method: RelevanceMethod,
) -> ComparisonPlotConfig:
    """
    Return the plot configuration for one method.
    """
    try:
        return PLOT_CONFIGS[
            method
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported relevance method: {method}"
        ) from error


def _get_class_name(
    class_id: int,
) -> str:
    """
    Return the readable name of one class.
    """
    try:
        class_index = list(
            CLASS_LABELS
        ).index(
            class_id
        )
    except ValueError as error:
        raise ValueError(
            f"Unknown class label: {class_id}"
        ) from error

    return CLASS_NAMES[
        class_index
    ]


def _validate_channel_names(
    channel_names: list[str],
) -> None:
    """
    Validate the EEG channel-name list.
    """
    if not channel_names:
        raise ValueError(
            "At least one channel name is required."
        )

    if len(channel_names) != len(
        set(channel_names)
    ):
        raise ValueError(
            "Channel names must be unique."
        )


def _validate_times(
    times: np.ndarray,
) -> None:
    """
    Validate a temporal axis.
    """
    if times.ndim != 1:
        raise ValueError(
            "times must be one-dimensional."
        )

    if len(times) == 0:
        raise ValueError(
            "times cannot be empty."
        )

    if not np.all(
        np.isfinite(
            times
        )
    ):
        raise ValueError(
            "times contains non-finite values."
        )

    if np.any(
        np.diff(
            times
        ) <= 0
    ):
        raise ValueError(
            "times must be strictly increasing."
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

    if window_end <= window_start:
        raise ValueError(
            "The imagery-window end must be "
            "greater than its start."
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