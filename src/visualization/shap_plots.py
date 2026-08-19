from collections.abc import Mapping
from typing import Literal

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.figure import Figure

from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)


TrialSelection = Literal[
    "correct",
    "incorrect",
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

FrequencyBand = tuple[float, float]


def plot_temporal_relevance(
    temporal_relevance: VectorRelevance,
    times: np.ndarray,
    trial_selection: TrialSelection,
    subject: int | None,
    imagery_window: tuple[float, float] = (0.5, 4.0),
    trial_counts: TrialCounts | None = None,
    ymin: float = 0.0,
    ymax: float | None = None,
) -> Figure:
    """
    Plot class-wise temporal EEGNet SHAP relevance.

    A shared ymin/ymax can be supplied so correct and incorrect
    trial selections use exactly the same vertical scale.
    """
    class_ids = _get_class_ids(
        temporal_relevance
    )

    _validate_class_relevance(
        class_ids
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    if times.ndim != 1:
        raise ValueError(
            "times must be one-dimensional."
        )

    # ----------------------------------------------------------
    # Y-axis scale
    # ----------------------------------------------------------

    if ymax is None:
        finite_values = []

        for class_id in class_ids:
            relevance = np.asarray(
                temporal_relevance[
                    class_id
                ],
                dtype=np.float64,
            )

            values = relevance[
                np.isfinite(
                    relevance
                )
            ]

            if values.size > 0:
                finite_values.append(
                    values
                )

        if finite_values:
            ymax = float(
                np.max(
                    np.concatenate(
                        finite_values
                    )
                )
            )
        else:
            ymax = float(
                np.finfo(float).eps
            )

    ymax = max(
        float(ymax),
        float(
            np.finfo(float).eps
        ),
    )

    if ymax <= ymin:
        raise ValueError(
            "ymax must be greater than ymin."
        )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    for class_id in class_ids:
        relevance = np.asarray(
            temporal_relevance[
                class_id
            ],
            dtype=np.float64,
        )

        if relevance.shape != times.shape:
            raise ValueError(
                "Each temporal relevance array must "
                "have the same shape as times."
            )

        axis.plot(
            times,
            relevance,
            label=_get_class_title(
                class_id,
                trial_counts,
            ),
        )

    # ----------------------------------------------------------
    # Shared axis limits
    # ----------------------------------------------------------

    window_start, window_end = (
        imagery_window
    )

    axis.set_xlim(
        window_start,
        window_end,
    )

    axis.set_ylim(
        ymin,
        ymax,
    )

    # ----------------------------------------------------------
    # Labels
    # ----------------------------------------------------------

    axis.set_xlabel(
        "Time relative to cue (s)"
    )

    axis.set_ylabel(
        "Mean absolute SHAP value"
    )

    axis.set_title(
        _build_title(
            base_title=(
                "EEGNet temporal SHAP relevance"
            ),
            trial_selection=trial_selection,
            subject=subject,
        )
    )

    axis.legend(
        title="Motor-imagery class"
    )

    axis.grid(
        alpha=0.25,
    )

    return figure


def plot_frequency_relevance(
    frequency_relevance: VectorRelevance,
    frequency_bands: tuple[
        tuple[float, float],
        ...,
    ],
    trial_selection: TrialSelection,
    subject: int | None,
    trial_counts: TrialCounts | None = None,
    ymin: float = 0.0,
    ymax: float | None = None,
) -> Figure:
    """
    Plot class-wise frequency-domain EEGNet SHAP relevance.

    A shared ymin/ymax can be supplied so correct and incorrect
    trial selections use exactly the same vertical scale.
    """
    class_ids = _get_class_ids(
        frequency_relevance
    )

    _validate_class_relevance(
        class_ids
    )

    n_bands = len(
        frequency_bands
    )

    if n_bands == 0:
        raise ValueError(
            "frequency_bands cannot be empty."
        )

    # ----------------------------------------------------------
    # Frequency-band labels
    # ----------------------------------------------------------

    band_labels = [
        f"{low:g}-{high:g}"
        for low, high
        in frequency_bands
    ]

    x_positions = np.arange(
        n_bands
    )

    # ----------------------------------------------------------
    # Shared Y-axis scale
    # ----------------------------------------------------------

    if ymax is None:
        finite_values = []

        for class_id in class_ids:
            relevance = np.asarray(
                frequency_relevance[
                    class_id
                ],
                dtype=np.float64,
            )

            values = relevance[
                np.isfinite(
                    relevance
                )
            ]

            if values.size > 0:
                finite_values.append(
                    values
                )

        if finite_values:
            ymax = float(
                np.max(
                    np.concatenate(
                        finite_values
                    )
                )
            )
        else:
            ymax = float(
                np.finfo(float).eps
            )

    ymax = max(
        float(ymax),
        float(
            np.finfo(float).eps
        ),
    )

    if ymax <= ymin:
        raise ValueError(
            "ymax must be greater than ymin."
        )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    for class_id in class_ids:
        relevance = np.asarray(
            frequency_relevance[
                class_id
            ],
            dtype=np.float64,
        )

        if relevance.shape != (
            n_bands,
        ):
            raise ValueError(
                "Each frequency relevance array must "
                "contain one value per frequency band."
            )

        axis.plot(
            x_positions,
            relevance,
            marker="o",
            label=_get_class_title(
                class_id,
                trial_counts,
            ),
        )

    # ----------------------------------------------------------
    # X-axis
    # ----------------------------------------------------------

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        band_labels,
        rotation=45,
        ha="right",
    )

    axis.set_xlabel(
        "Frequency band (Hz)"
    )

    # ----------------------------------------------------------
    # Shared Y-axis
    # ----------------------------------------------------------

    axis.set_ylim(
        ymin,
        ymax,
    )

    axis.set_ylabel(
        "Mean absolute SHAP value"
    )

    # ----------------------------------------------------------
    # Title
    # ----------------------------------------------------------

    axis.set_title(
        _build_title(
            base_title=(
                "EEGNet frequency-domain SHAP relevance"
            ),
            trial_selection=trial_selection,
            subject=subject,
        )
    )

    # ----------------------------------------------------------
    # Legend + grid
    # ----------------------------------------------------------

    axis.legend(
        title="Motor-imagery class",
        loc="upper right",
    )

    axis.grid(
        alpha=0.25,
    )

    return figure


def plot_topographies(
    topographic_relevance: VectorRelevance,
    info: mne.Info,
    trial_selection: TrialSelection,
    subject: int | None,
    imagery_window: tuple[float, float] = (0.5, 4.0),
    trial_counts: TrialCounts | None = None,
    show_channel_names: bool = False,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> Figure:
    """
    Plot class-wise scalp topographies of SHAP relevance.
    """
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

    if vmax is None:
        vmax = max(
            np.max(
                topographic_relevance[
                    class_id
                ]
            )
            for class_id in class_ids
        )

    vmax = max(
        float(vmax),
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
                vmin,
                vmax,
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
            "Mean absolute SHAP value"
        )

    window_start, window_end = (
        imagery_window
    )

    figure.suptitle(
        _build_title(
            base_title=(
                "EEGNet SHAP topographies "
                f"({window_start:.1f}–{window_end:.1f} s)"
            ),
            trial_selection=trial_selection,
            subject=subject,
        ),
        fontsize=15,
    )

    return figure


def plot_channel_relevance(
    channel_relevance: VectorRelevance,
    channel_names: list[str],
    trial_selection: TrialSelection,
    subject: int | None,
    trial_counts: TrialCounts | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> Figure:
    """
    Plot a class-by-channel SHAP relevance heatmap.

    A shared vmin/vmax can be supplied so correct and incorrect
    trial selections use the same color scale.
    """
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

    # ----------------------------------------------------------
    # Color scale
    # ----------------------------------------------------------

    if vmax is None:
        vmax = float(
            np.nanmax(
                relevance_matrix
            )
        )

    vmax = max(
        float(vmax),
        float(
            np.finfo(float).eps
        ),
    )

    if vmax <= vmin:
        raise ValueError(
            "vmax must be greater than vmin."
        )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    figure, axis = plt.subplots(
        figsize=(15, 5),
        constrained_layout=True,
    )

    image = axis.imshow(
        relevance_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
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
        _build_title(
            base_title=(
                "Class-wise EEG channel SHAP relevance"
            ),
            trial_selection=trial_selection,
            subject=subject,
        )
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
        "Mean absolute SHAP value"
    )

    return figure


def plot_channel_rankings(
    channel_rankings: ChannelRankings,
    trial_selection: TrialSelection,
    subject: int | None,
    top_n: int = 10,
    trial_counts: TrialCounts | None = None,
    xmax: float | None = None,
) -> Figure:
    """
    Plot the highest-ranked EEG channels for each class.

    A shared xmax can be supplied so correct and incorrect
    ranking plots use the same horizontal relevance scale.
    """
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

    # ----------------------------------------------------------
    # Shared x-axis scale
    # ----------------------------------------------------------

    if xmax is None:
        relevance_values = [
            float(relevance)
            for class_id in class_ids
            for _, relevance
            in channel_rankings[
                class_id
            ][:top_n]
        ]

        if relevance_values:
            xmax = max(
                max(relevance_values),
                float(
                    np.finfo(float).eps
                ),
            )
        else:
            xmax = float(
                np.finfo(float).eps
            )

    xmax = max(
        float(xmax),
        float(
            np.finfo(float).eps
        ),
    )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

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

        # ------------------------------------------------------
        # Same numerical scale for every class and for
        # correct/incorrect figures.
        # ------------------------------------------------------

        axis.set_xlim(
            0.0,
            xmax,
        )

        axis.set_title(
            _get_class_title(
                class_id,
                trial_counts,
            )
        )

        axis.set_xlabel(
            "Mean absolute SHAP value"
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
        _build_title(
            base_title=(
                f"Top {top_n} EEG channels "
                "by SHAP relevance"
            ),
            trial_selection=trial_selection,
            subject=subject,
        ),
        fontsize=15,
    )

    return figure


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


def _get_trial_selection_label(
    trial_selection: TrialSelection,
) -> str:
    """
    Return the readable label for one trial subset.
    """
    labels = {
        "correct": "correct trials",
        "incorrect": "incorrect trials",
    }

    try:
        return labels[
            trial_selection
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported trial selection: {trial_selection}"
        ) from error


def _get_result_scope_label(
    subject: int | None,
) -> str:
    """
    Return whether the result is subject-wise or mean across subjects.
    """
    if subject is None:
        return "mean across subjects"

    return _get_subject_name(
        subject
    )


def _build_title(
    base_title: str,
    trial_selection: TrialSelection,
    subject: int | None,
) -> str:
    """
    Build a title including trial subset and subject/global scope.
    """
    trial_label = _get_trial_selection_label(
        trial_selection
    )

    scope_label = _get_result_scope_label(
        subject
    )

    return (
        f"{base_title}\n"
        f"({trial_label}, {scope_label})"
    )


def _get_subject_name(
    subject: int,
) -> str:
    """
    Return the formatted subject name.
    """
    return f"A{subject:02d}"