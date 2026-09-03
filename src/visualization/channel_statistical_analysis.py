import matplotlib.pyplot as plt
import numpy as np

from src.analysis.channel_statistical_analysis import (
    COMPARISONS,
    ChannelStatisticRow,
)
from src.utils.paths import (
    get_channel_statistical_comparison_plot_path,
)


def plot_channel_statistical_summaries(
    rows: list[ChannelStatisticRow],
) -> list:
    """
    Plot one channel relevance summary per comparison.
    """
    output_paths = []

    for comparison in COMPARISONS:
        comparison_rows = [
            row
            for row in rows
            if row.comparison_slug == comparison.slug
        ]

        if not comparison_rows:
            continue

        output_path = get_channel_statistical_comparison_plot_path(
            comparison.slug
        )

        _plot_comparison(
            rows=comparison_rows,
            title=comparison.name,
            left_label=comparison.left_label,
            right_label=comparison.right_label,
            output_path=output_path,
        )

        output_paths.append(
            output_path
        )

    return output_paths


def _plot_comparison(
    rows: list[ChannelStatisticRow],
    title: str,
    left_label: str,
    right_label: str,
    output_path,
) -> None:
    """
    Create one channel relevance comparison plot.
    """
    x_positions = np.arange(
        len(rows)
    )

    left_means = np.asarray(
        [
            row.mean_a
            for row in rows
        ],
        dtype=np.float64,
    )

    right_means = np.asarray(
        [
            row.mean_b
            for row in rows
        ],
        dtype=np.float64,
    )

    left_stds = np.asarray(
        [
            row.std_a
            for row in rows
        ],
        dtype=np.float64,
    )

    right_stds = np.asarray(
        [
            row.std_b
            for row in rows
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(15, 7),
        constrained_layout=True,
    )

    axis.errorbar(
        x_positions,
        left_means,
        yerr=left_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=left_label,
    )

    axis.errorbar(
        x_positions,
        right_means,
        yerr=right_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=right_label,
    )

    finite_upper = np.concatenate([
        left_means + np.nan_to_num(
            left_stds,
            nan=0.0,
        ),
        right_means + np.nan_to_num(
            right_stds,
            nan=0.0,
        ),
    ])

    y_max = float(
        np.nanmax(
            finite_upper
        )
    )

    y_max = max(
        y_max,
        float(
            np.finfo(float).eps
        ),
    )

    axis.set_ylim(
        0.0,
        y_max * 1.18,
    )

    for index, row in enumerate(
        rows
    ):
        if not row.significant_fdr:
            continue

        axis.text(
            index,
            y_max * 1.08,
            "*",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            row.channel
            for row in rows
        ],
        rotation=45,
        ha="right",
    )

    axis.set_xlabel(
        "EEG channel"
    )

    axis.set_ylabel(
        "Mean relative channel relevance"
    )

    axis.set_title(
        title
    )

    axis.legend()

    axis.grid(
        alpha=0.25,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )
