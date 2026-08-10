from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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