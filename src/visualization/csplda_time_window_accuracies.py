from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot_csplda_time_window_accuracy_comparison(
    time_windows: list[tuple[float, float]],
    mean_window_accuracies: list[float],
    mean_baseline_accuracy: float,
    mean_eegnet_accuracy: float,
    output_path: Path,
) -> None:
    """
    Plot mean CSP+LDA accuracy for different temporal windows.

    Each bar shows the mean CSP+LDA accuracy across subjects
    for one temporal window.

    Horizontal reference lines show the mean baseline CSP+LDA
    accuracy and the mean EEGNet accuracy.
    """
    if len(time_windows) != len(mean_window_accuracies):
        raise ValueError(
            "time_windows and mean_window_accuracies "
            "must have the same length."
        )

    labels = [
        f"{tmin:.1f}-{tmax:.1f}"
        for tmin, tmax in time_windows
    ]

    accuracies = np.asarray(
        mean_window_accuracies,
        dtype=float,
    )

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(11, 6),
    )

    bars = ax.bar(
        x,
        accuracies,
        width=0.65,
    )

    ax.axhline(
        mean_baseline_accuracy,
        linestyle="--",
        linewidth=1.5,
        color="dimgray",
        label=(
            "CSP+LDA baseline "
            f"({mean_baseline_accuracy * 100:.2f}%)"
        ),
    )

    ax.axhline(
        mean_eegnet_accuracy,
        linestyle="--",
        linewidth=1.5,
        color="crimson",
        label=(
            "EEGNet "
            f"({mean_eegnet_accuracy * 100:.2f}%)"
        ),
    )

    for bar, accuracy in zip(
        bars,
        accuracies,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() - 0.015,
            f"{accuracy * 100:.2f}%",
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.set_title(
        "CSP+LDA Accuracy Across Temporal Windows"
    )

    ax.set_xlabel(
        "Temporal window (s)"
    )

    ax.set_ylabel(
        "Mean accuracy (%)"
    )

    ax.set_xticks(
        x,
    )

    ax.set_xticklabels(
        labels,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="upper left",
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)