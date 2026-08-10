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
    output_dir: Path,
) -> None:
    """
    Plot class-wise CSP+LDA channel relevance.
    """
    channel_relevance = np.asarray(
        channel_relevance,
        dtype=np.float64,
    )

    if channel_relevance.shape != (
        len(class_names),
        len(channel_names),
    ):
        raise ValueError(
            "channel_relevance must have shape "
            "(n_classes, n_channels)."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(14, 5),
    )

    image = ax.imshow(
        channel_relevance,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(channel_names)),
    )
    ax.set_xticklabels(
        channel_names,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        np.arange(len(class_names)),
    )
    ax.set_yticklabels(
        class_names,
    )

    ax.set_xlabel("EEG channel")
    ax.set_ylabel("Motor-imagery class")

    ax.set_title(
        "Class-wise CSP+LDA channel relevance\n"
        f"({subject})"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        pad=0.04,
    )

    colorbar.set_label(
        "Normalized channel relevance"
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / f"{subject}_csp_channel_relevance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_csp_channel_rankings(
    channel_relevance: np.ndarray,
    channel_names: list[str],
    class_names: list[str],
    subject: str,
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """
    Plot the most relevant EEG channels for each motor-imagery class.
    """
    channel_relevance = np.asarray(
        channel_relevance,
        dtype=np.float64,
    )

    if channel_relevance.shape != (
        len(class_names),
        len(channel_names),
    ):
        raise ValueError(
            "channel_relevance must have shape "
            "(n_classes, n_channels)."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_classes = len(class_names)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axes = axes.flatten()

    for class_idx in range(n_classes):
        ax = axes[class_idx]

        relevance = channel_relevance[
            class_idx
        ]

        ranking = np.argsort(
            relevance
        )[::-1][:top_n]

        ranked_channels = [
            channel_names[index]
            for index in ranking
        ]

        ranked_relevance = relevance[
            ranking
        ]

        # Reverse so that the most relevant
        # channel appears at the top.
        ranked_channels = ranked_channels[::-1]
        ranked_relevance = ranked_relevance[::-1]

        ax.barh(
            ranked_channels,
            ranked_relevance,
        )

        ax.set_title(
            class_names[class_idx]
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

    fig.suptitle(
        "Top 10 EEG channels by CSP+LDA relevance\n"
        f"({subject})",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / f"{subject}_csp_channel_rankings.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_csp_temporal_relevance(
    temporal_relevance: np.ndarray,
    times: np.ndarray,
    class_names: list[str],
    subject: str,
    output_dir: Path,
) -> None:
    """
    Plot class-wise CSP+LDA temporal relevance.
    """
    temporal_relevance = np.asarray(
        temporal_relevance,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    if temporal_relevance.ndim != 2:
        raise ValueError(
            "temporal_relevance must have shape "
            "(n_classes, n_times)."
        )

    if temporal_relevance.shape[0] != len(
        class_names
    ):
        raise ValueError(
            "Number of classes in temporal_relevance "
            "does not match class_names."
        )

    if temporal_relevance.shape[1] != len(
        times
    ):
        raise ValueError(
            "Number of time points in temporal_relevance "
            "does not match times."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for class_idx, class_name in enumerate(
        class_names
    ):
        ax = axes[class_idx]

        ax.plot(
            times,
            temporal_relevance[class_idx],
        )

        ax.set_title(
            class_name
        )

        ax.set_xlabel(
            "Time relative to cue (s)"
        )

        ax.set_ylabel(
            "Normalized temporal relevance"
        )

        ax.grid(
            alpha=0.25,
        )

    if subject == "all_mean":
        title_suffix = "mean across subjects"
    else:
        title_suffix = subject

    fig.suptitle(
        "CSP+LDA temporal relevance\n"
        f"({title_suffix})",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / f"{subject}_csp_temporal_relevance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_csp_frequency_relevance(
    frequency_relevance: np.ndarray,
    frequencies: np.ndarray,
    class_names: list[str],
    subject: str,
    output_dir: Path,
) -> None:
    """
    Plot class-wise CSP+LDA frequency relevance.
    """
    frequency_relevance = np.asarray(
        frequency_relevance,
        dtype=np.float64,
    )

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    if frequency_relevance.ndim != 2:
        raise ValueError(
            "frequency_relevance must have shape "
            "(n_classes, n_frequencies)."
        )

    if frequency_relevance.shape[0] != len(
        class_names
    ):
        raise ValueError(
            "Number of classes in frequency_relevance "
            "does not match class_names."
        )

    if frequency_relevance.shape[1] != len(
        frequencies
    ):
        raise ValueError(
            "Number of frequency points in "
            "frequency_relevance does not match frequencies."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for class_idx, class_name in enumerate(
        class_names
    ):
        ax = axes[class_idx]

        ax.plot(
            frequencies,
            frequency_relevance[class_idx],
        )

        ax.set_title(
            class_name
        )

        ax.set_xlabel(
            "Frequency (Hz)"
        )

        ax.set_ylabel(
            "Normalized frequency relevance"
        )

        ax.grid(
            alpha=0.25,
        )

    if subject == "all_mean":
        title_suffix = "mean across subjects"
    else:
        title_suffix = subject

    fig.suptitle(
        "CSP+LDA frequency relevance\n"
        f"({title_suffix})",
        fontsize=16,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        / f"{subject}_csp_frequency_relevance.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_csp_topographies(
    channel_relevance: np.ndarray,
    channel_names: list[str],
    class_names: list[str],
    subject: str,
    output_dir: Path,
    sfreq: float = 250.0,
) -> None:
    """
    Plot class-wise CSP+LDA channel-relevance topographies.
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
    # Shared scale across all four classes
    # ----------------------------------------------------------

    vmin = 0.0

    vmax = float(
        np.max(channel_relevance)
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

        mne.viz.plot_topomap(
            channel_relevance[class_idx],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="viridis",
            vlim=(vmin, vmax),
            sensors=True,
        )

        ax.set_title(
            class_name,
            fontsize=13,
            pad=10,
        )

    # ----------------------------------------------------------
    # Title
    # ----------------------------------------------------------

    if subject == "all_mean":
        title_suffix = "mean across subjects"
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

    plt.close(fig)