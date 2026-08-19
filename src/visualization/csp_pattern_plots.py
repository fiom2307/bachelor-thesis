from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_csp_channel_relevance(
    channel_relevance: np.ndarray,
    channel_names: list[str],
    class_names: list[str],
    subject: str,
    output_dir,
    class_counts: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> None:
    """
    Plot class-wise CSP+LDA channel relevance as a heatmap.

    A shared vmin/vmax can be supplied so correct and incorrect
    trial selections use the same color scale.
    """
    channel_relevance = np.asarray(
        channel_relevance,
        dtype=np.float64,
    )

    expected_shape = (
        len(class_names),
        len(channel_names),
    )

    if channel_relevance.shape != expected_shape:
        raise ValueError(
            "channel_relevance must have shape "
            f"{expected_shape}."
        )

    if class_counts is not None:
        class_counts = np.asarray(
            class_counts,
            dtype=int,
        )

        if class_counts.shape != (
            len(class_names),
        ):
            raise ValueError(
                "class_counts must contain one "
                "value per class."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Color scale
    # ----------------------------------------------------------

    if vmax is None:
        finite_values = channel_relevance[
            np.isfinite(
                channel_relevance
            )
        ]

        if finite_values.size == 0:
            vmax = float(
                np.finfo(float).eps
            )
        else:
            vmax = float(
                np.max(
                    finite_values
                )
            )

    vmax = max(
        float(vmax),
        float(
            np.finfo(float).eps
        ),
    )

    # ----------------------------------------------------------
    # Plot
    # ----------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(15, 5),
        constrained_layout=True,
    )

    masked_relevance = np.ma.masked_invalid(
        channel_relevance
    )

    image = ax.imshow(
        masked_relevance,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(
        np.arange(
            len(channel_names)
        )
    )

    ax.set_xticklabels(
        channel_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(
            len(class_names)
        )
    )

    if class_counts is None:
        y_labels = class_names
    else:
        y_labels = [
            f"{class_name} "
            f"(n={class_counts[class_idx]})"
            for class_idx, class_name
            in enumerate(class_names)
        ]

    ax.set_yticklabels(
        y_labels
    )

    ax.set_xlabel(
        "EEG channel"
    )

    ax.set_ylabel(
        "Motor-imagery class"
    )

    # ----------------------------------------------------------
    # Title
    # ----------------------------------------------------------

    if subject == "all_mean_correct":
        title_suffix = (
            "correct trials, mean across subjects"
        )

    elif subject == "all_mean_incorrect":
        title_suffix = (
            "incorrect trials, mean across subjects"
        )

    elif subject.endswith(
        "_correct"
    ):
        title_suffix = (
            "correct trials, "
            f"{subject.removesuffix('_correct')}"
        )

    elif subject.endswith(
        "_incorrect"
    ):
        title_suffix = (
            "incorrect trials, "
            f"{subject.removesuffix('_incorrect')}"
        )

    else:
        title_suffix = subject

    ax.set_title(
        "CSP+LDA channel relevance\n"
        f"({title_suffix})",
        fontsize=15,
    )

    # ----------------------------------------------------------
    # Colorbar
    # ----------------------------------------------------------

    colorbar = fig.colorbar(
        image,
        ax=ax,
        shrink=0.9,
    )

    colorbar.set_label(
        "Normalized channel relevance"
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_path = (
        output_dir
        / f"{subject}_csp_channel_relevance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_csp_channel_rankings(
    channel_relevance: np.ndarray,
    channel_names: list[str],
    class_names: list[str],
    subject: str,
    output_dir,
    top_n: int = 10,
    class_counts: np.ndarray | None = None,
    xmax: float | None = None,
) -> None:
    """
    Plot the highest-ranked CSP+LDA channels for each class.

    A shared xmax can be supplied so correct and incorrect
    ranking figures use the same relevance scale.
    """
    channel_relevance = np.asarray(
        channel_relevance,
        dtype=np.float64,
    )

    expected_shape = (
        len(class_names),
        len(channel_names),
    )

    if channel_relevance.shape != expected_shape:
        raise ValueError(
            "channel_relevance must have shape "
            f"{expected_shape}."
        )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    if class_counts is not None:
        class_counts = np.asarray(
            class_counts,
            dtype=int,
        )

        if class_counts.shape != (
            len(class_names),
        ):
            raise ValueError(
                "class_counts must contain one "
                "value per class."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Shared x-axis scale
    # ----------------------------------------------------------

    if xmax is None:
        finite_values = channel_relevance[
            np.isfinite(
                channel_relevance
            )
        ]

        if finite_values.size == 0:
            xmax = float(
                np.finfo(float).eps
            )
        else:
            xmax = float(
                np.max(
                    finite_values
                )
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

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        constrained_layout=True,
    )

    axes = axes.flatten()

    for class_idx, class_name in enumerate(
        class_names
    ):
        ax = axes[
            class_idx
        ]

        relevance = channel_relevance[
            class_idx
        ]

        count = (
            None
            if class_counts is None
            else int(
                class_counts[
                    class_idx
                ]
            )
        )

        # ------------------------------------------------------
        # Title
        # ------------------------------------------------------

        if count is None:
            class_title = class_name
        else:
            class_title = (
                f"{class_name} "
                f"(n={count})"
            )

        ax.set_title(
            class_title
        )

        ax.set_xlim(
            0.0,
            xmax,
        )

        # ------------------------------------------------------
        # No selected trials
        # ------------------------------------------------------

        finite_mask = np.isfinite(
            relevance
        )

        if (
            count == 0
            or not np.any(
                finite_mask
            )
        ):
            ax.text(
                0.5,
                0.5,
                "No trials",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

            ax.set_xlabel(
                "Normalized channel relevance"
            )

            ax.set_yticks([])

            continue

        # ------------------------------------------------------
        # Ranking
        # ------------------------------------------------------

        valid_indices = np.where(
            finite_mask
        )[0]

        valid_relevance = relevance[
            finite_mask
        ]

        ranking_order = np.argsort(
            valid_relevance
        )[::-1]

        ranking_order = ranking_order[
            :top_n
        ]

        ranked_indices = valid_indices[
            ranking_order
        ]

        ranked_values = relevance[
            ranked_indices
        ]

        ranked_names = [
            channel_names[
                channel_idx
            ]
            for channel_idx
            in ranked_indices
        ]

        # Reverse so highest channel appears at the top.
        ranked_names = (
            ranked_names[::-1]
        )

        ranked_values = (
            ranked_values[::-1]
        )

        ax.barh(
            ranked_names,
            ranked_values,
        )

        ax.set_xlim(
            0.0,
            xmax,
        )

        ax.set_xlabel(
            "Normalized channel relevance"
        )

        ax.set_ylabel(
            "EEG channel"
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

    # ----------------------------------------------------------
    # Main title
    # ----------------------------------------------------------

    if subject == "all_mean_correct":
        title_suffix = (
            "correct trials, mean across subjects"
        )

    elif subject == "all_mean_incorrect":
        title_suffix = (
            "incorrect trials, mean across subjects"
        )

    elif subject.endswith(
        "_correct"
    ):
        title_suffix = (
            "correct trials, "
            f"{subject.removesuffix('_correct')}"
        )

    elif subject.endswith(
        "_incorrect"
    ):
        title_suffix = (
            "incorrect trials, "
            f"{subject.removesuffix('_incorrect')}"
        )

    else:
        title_suffix = subject

    fig.suptitle(
        f"Top {top_n} EEG channels "
        "by CSP+LDA relevance\n"
        f"({title_suffix})",
        fontsize=15,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_path = (
        output_dir
        / f"{subject}_csp_channel_rankings.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_csp_temporal_relevance(
    temporal_relevance: np.ndarray,
    times: np.ndarray,
    class_names: list[str],
    subject: str,
    output_dir,
    class_counts: np.ndarray | None = None,
    ymin: float = 0.0,
    ymax: float | None = None,
) -> None:
    """
    Plot class-wise CSP+LDA temporal relevance.

    A shared ymin/ymax can be supplied so correct and incorrect
    trial selections use exactly the same vertical scale.
    """
    temporal_relevance = np.asarray(
        temporal_relevance,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    expected_shape = (
        len(class_names),
        len(times),
    )

    if temporal_relevance.shape != expected_shape:
        raise ValueError(
            "temporal_relevance must have shape "
            f"{expected_shape}."
        )

    if times.ndim != 1:
        raise ValueError(
            "times must be one-dimensional."
        )

    if class_counts is not None:
        class_counts = np.asarray(
            class_counts,
            dtype=int,
        )

        if class_counts.shape != (
            len(class_names),
        ):
            raise ValueError(
                "class_counts must contain one "
                "value per class."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Shared Y-axis scale
    # ----------------------------------------------------------

    if ymax is None:
        finite_values = temporal_relevance[
            np.isfinite(
                temporal_relevance
            )
        ]

        if finite_values.size == 0:
            ymax = float(
                np.finfo(float).eps
            )
        else:
            ymax = float(
                np.max(
                    finite_values
                )
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

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    for class_idx, class_name in enumerate(
        class_names
    ):
        relevance = temporal_relevance[
            class_idx
        ]

        # Skip a completely unavailable class.
        if not np.any(
            np.isfinite(
                relevance
            )
        ):
            continue

        if class_counts is None:
            label = class_name
        else:
            label = (
                f"{class_name} "
                f"(n={class_counts[class_idx]})"
            )

        ax.plot(
            times,
            relevance,
            label=label,
        )

    # ----------------------------------------------------------
    # Axis limits
    # ----------------------------------------------------------

    ax.set_xlim(
        times[0],
        times[-1],
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    # ----------------------------------------------------------
    # Labels
    # ----------------------------------------------------------

    ax.set_xlabel(
        "Time relative to cue (s)"
    )

    ax.set_ylabel(
        "Normalized temporal relevance"
    )

    # ----------------------------------------------------------
    # Title formatting, same style as SHAP
    # ----------------------------------------------------------

    if subject == "all_mean_correct":
        title_suffix = (
            "correct trials, mean across subjects"
        )

    elif subject == "all_mean_incorrect":
        title_suffix = (
            "incorrect trials, mean across subjects"
        )

    elif subject.endswith(
        "_correct"
    ):
        title_suffix = (
            "correct trials, "
            f"{subject.removesuffix('_correct')}"
        )

    elif subject.endswith(
        "_incorrect"
    ):
        title_suffix = (
            "incorrect trials, "
            f"{subject.removesuffix('_incorrect')}"
        )

    else:
        title_suffix = subject

    ax.set_title(
        "CSP+LDA temporal relevance\n"
        f"({title_suffix})"
    )

    # ----------------------------------------------------------
    # Legend + grid
    # ----------------------------------------------------------

    ax.legend(
        title="Motor-imagery class",
        loc="upper right",
    )

    ax.grid(
        alpha=0.25,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_path = (
        output_dir
        / f"{subject}_csp_temporal_relevance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_csp_frequency_relevance(
    frequency_relevance: np.ndarray,
    frequencies: np.ndarray,
    class_names: list[str],
    subject: str,
    output_dir,
    class_counts: np.ndarray | None = None,
    ymin: float = 0.0,
    ymax: float | None = None,
) -> None:
    """
    Plot class-wise CSP+LDA frequency relevance.

    A shared ymin/ymax can be supplied so correct and incorrect
    trial selections use exactly the same vertical scale.
    """
    frequency_relevance = np.asarray(
        frequency_relevance,
        dtype=np.float64,
    )

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    expected_shape = (
        len(class_names),
        len(frequencies),
    )

    if frequency_relevance.shape != expected_shape:
        raise ValueError(
            "frequency_relevance must have shape "
            f"{expected_shape}."
        )

    if frequencies.ndim != 1:
        raise ValueError(
            "frequencies must be one-dimensional."
        )

    if class_counts is not None:
        class_counts = np.asarray(
            class_counts,
            dtype=int,
        )

        if class_counts.shape != (
            len(class_names),
        ):
            raise ValueError(
                "class_counts must contain one "
                "value per class."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # Shared Y-axis scale
    # ----------------------------------------------------------

    if ymax is None:
        finite_values = frequency_relevance[
            np.isfinite(
                frequency_relevance
            )
        ]

        if finite_values.size == 0:
            ymax = float(
                np.finfo(float).eps
            )
        else:
            ymax = float(
                np.max(
                    finite_values
                )
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

    fig, ax = plt.subplots(
        figsize=(13, 7),
        constrained_layout=True,
    )

    for class_idx, class_name in enumerate(
        class_names
    ):
        relevance = frequency_relevance[
            class_idx
        ]

        finite_mask = np.isfinite(
            relevance
        )

        if not np.any(
            finite_mask
        ):
            continue

        if class_counts is None:
            label = class_name
        else:
            label = (
                f"{class_name} "
                f"(n={class_counts[class_idx]})"
            )

        ax.plot(
            frequencies[
                finite_mask
            ],
            relevance[
                finite_mask
            ],
            label=label,
        )

    # ----------------------------------------------------------
    # Axes
    # ----------------------------------------------------------

    ax.set_xlim(
        frequencies[0],
        frequencies[-1],
    )

    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_xlabel(
        "Frequency (Hz)"
    )

    ax.set_ylabel(
        "Normalized frequency relevance"
    )

    # ----------------------------------------------------------
    # Title — same structure as SHAP
    # ----------------------------------------------------------

    if subject == "all_mean_correct":
        title_suffix = (
            "correct trials, mean across subjects"
        )

    elif subject == "all_mean_incorrect":
        title_suffix = (
            "incorrect trials, mean across subjects"
        )

    elif subject.endswith(
        "_correct"
    ):
        title_suffix = (
            "correct trials, "
            f"{subject.removesuffix('_correct')}"
        )

    elif subject.endswith(
        "_incorrect"
    ):
        title_suffix = (
            "incorrect trials, "
            f"{subject.removesuffix('_incorrect')}"
        )

    else:
        title_suffix = subject

    ax.set_title(
        "CSP+LDA frequency relevance\n"
        f"({title_suffix})"
    )

    # ----------------------------------------------------------
    # Legend + grid
    # ----------------------------------------------------------

    ax.legend(
        title="Motor-imagery class",
        loc="upper right",
    )

    ax.grid(
        alpha=0.25,
    )

    # ----------------------------------------------------------
    # Frequency ticks
    #
    # Keep them readable and similar to the SHAP frequency plot.
    # ----------------------------------------------------------

    ax.set_xticks(
        np.arange(
            8,
            31,
            2,
        )
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_path = (
        output_dir
        / f"{subject}_csp_frequency_relevance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def plot_csp_topographies(
    channel_relevance: np.ndarray,
    channel_names: list[str],
    class_names: list[str],
    subject: str,
    output_dir,
    sfreq: float,
    class_counts: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
) -> None:
    """
    Plot class-wise CSP+LDA channel-relevance topographies.

    A shared vmin/vmax can be supplied so correct and incorrect
    trial selections use the same color scale.
    """
    channel_relevance = np.asarray(
        channel_relevance,
        dtype=np.float64,
    )

    expected_shape = (
        len(class_names),
        len(channel_names),
    )

    if channel_relevance.shape != expected_shape:
        raise ValueError(
            "channel_relevance must have shape "
            f"{expected_shape}."
        )

    if class_counts is not None:
        class_counts = np.asarray(
            class_counts,
            dtype=int,
        )

        if class_counts.shape != (
            len(class_names),
        ):
            raise ValueError(
                "class_counts must contain one "
                "value per class."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # MNE channel information
    # ----------------------------------------------------------

    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    montage = mne.channels.make_standard_montage(
        "standard_1020"
    )

    info.set_montage(
        montage,
        match_case=False,
        on_missing="warn",
    )

    # ----------------------------------------------------------
    # Color scale
    #
    # If vmax is supplied by the analysis script, use it.
    # Otherwise fall back to the maximum of this plot.
    # ----------------------------------------------------------

    if vmax is None:
        vmax = float(
            np.nanmax(
                channel_relevance
            )
        )

    vmax = max(
        float(vmax),
        float(
            np.finfo(float).eps
        ),
    )

    # ----------------------------------------------------------
    # Figure
    # ----------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 9),
    )

    axes = axes.flatten()

    for class_idx, class_name in enumerate(
        class_names
    ):
        ax = axes[class_idx]

        relevance = (
            channel_relevance[
                class_idx
            ]
        )

        mne.viz.plot_topomap(
            relevance,
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="viridis",
            vlim=(
                vmin,
                vmax,
            ),
            sensors=True,
        )

        # ------------------------------------------------------
        # Class title with trial count
        # ------------------------------------------------------

        if class_counts is not None:
            class_title = (
                f"{class_name} "
                f"(n={class_counts[class_idx]})"
            )
        else:
            class_title = class_name

        ax.set_title(
            class_title,
            fontsize=13,
            pad=10,
        )

    # ----------------------------------------------------------
    # Title
    # ----------------------------------------------------------

    if subject == "all_mean":
        title_suffix = (
            "mean across subjects"
        )
    else:
        title_suffix = subject

    fig.suptitle(
        "CSP+LDA channel relevance topographies\n"
        f"({title_suffix})",
        fontsize=16,
        y=0.97,
    )

    # Leave explicit room on the right for colorbar.
    fig.subplots_adjust(
        left=0.05,
        right=0.87,
        bottom=0.06,
        top=0.88,
        wspace=0.20,
        hspace=0.30,
    )

    # ----------------------------------------------------------
    # Separate colorbar axis
    # ----------------------------------------------------------

    colorbar_ax = fig.add_axes(
        [
            0.90,
            0.20,
            0.025,
            0.60,
        ]
    )

    norm = Normalize(
        vmin=vmin,
        vmax=vmax,
    )

    scalar_mappable = ScalarMappable(
        norm=norm,
        cmap="viridis",
    )

    scalar_mappable.set_array([])

    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
    )

    colorbar.set_label(
        "Normalized channel relevance"
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_path = (
        output_dir
        / f"{subject}_csp_topographies.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )