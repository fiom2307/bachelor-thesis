from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot_csplda_n_components_accuracy_comparison(
    n_components_list: list[int],
    mean_component_accuracies: list[float],
    mean_baseline_accuracy: float,
    mean_eegnet_accuracy: float,
    output_path: Path,
) -> None:
    """
    Plot mean CSP+LDA accuracy for different numbers of CSP components.

    Each bar shows the mean CSP+LDA accuracy across subjects for one
    component configuration. Horizontal reference lines show the mean
    baseline CSP+LDA accuracy and the mean EEGNet accuracy.
    """
    if len(n_components_list) != len(mean_component_accuracies):
        raise ValueError(
            "n_components_list and mean_component_accuracies "
            "must have the same length."
        )

    accuracies = np.asarray(
        mean_component_accuracies,
        dtype=float,
    )

    x = np.arange(
        len(n_components_list)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6),
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
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accuracy * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title(
        "CSP+LDA Accuracy Across Numbers of CSP Components"
    )

    ax.set_xlabel(
        "Number of CSP components"
    )

    ax.set_ylabel(
        "Mean accuracy (%)"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            str(n_components)
            for n_components in n_components_list
        ]
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
