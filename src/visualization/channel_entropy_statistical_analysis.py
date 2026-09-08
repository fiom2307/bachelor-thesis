import matplotlib.pyplot as plt
import numpy as np

from src.analysis.channel_entropy_statistical_analysis import (
    ChannelEntropyStatisticRow,
)
from src.analysis.channel_statistical_analysis import COMPARISONS
from src.data.labels import CLASS_NAMES
from src.utils.paths import (
    get_channel_entropy_statistical_classwise_plot_path,
    get_channel_entropy_statistical_overall_plot_path,
)
from src.visualization.common import save_figure


def plot_channel_entropy_statistical_summaries(
    rows_by_output: dict[str, list[ChannelEntropyStatisticRow]],
) -> list:
    """
    Plot overall.png and classwise.png per channel entropy comparison.
    """
    output_paths = []

    for comparison in COMPARISONS:
        overall_rows = rows_by_output.get(
            f"{comparison.slug}:overall",
            [],
        )

        if overall_rows:
            output_path = get_channel_entropy_statistical_overall_plot_path(
                comparison.slug
            )
            _plot_overall_entropy(
                row=overall_rows[0],
                title=f"{comparison.name} channel entropy",
                left_label=comparison.left_label,
                right_label=comparison.right_label,
                output_path=output_path,
            )
            output_paths.append(
                output_path
            )

        classwise_rows = rows_by_output.get(
            f"{comparison.slug}:classwise",
            [],
        )

        if classwise_rows:
            output_path = get_channel_entropy_statistical_classwise_plot_path(
                comparison.slug
            )
            _plot_classwise_entropy(
                rows=classwise_rows,
                title=f"{comparison.name} channel entropy",
                left_label=comparison.left_label,
                right_label=comparison.right_label,
                output_path=output_path,
            )
            output_paths.append(
                output_path
            )

    return output_paths


def _plot_overall_entropy(
    row: ChannelEntropyStatisticRow,
    title: str,
    left_label: str,
    right_label: str,
    output_path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(5, 5),
        constrained_layout=True,
    )

    _plot_entropy_rows_on_axis(
        axis=axis,
        rows=[row],
        x_labels=["Overall"],
        left_label=left_label,
        right_label=right_label,
    )
    axis.set_title(
        title
    )
    axis.legend()

    save_figure(
        figure,
        output_path,
    )
    plt.close(
        figure
    )


def _plot_classwise_entropy(
    rows: list[ChannelEntropyStatisticRow],
    title: str,
    left_label: str,
    right_label: str,
    output_path,
) -> None:
    ordered_rows = [
        next(
            row
            for row in rows
            if row.class_name == class_name
        )
        for class_name in CLASS_NAMES
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    _plot_entropy_rows_on_axis(
        axis=axis,
        rows=ordered_rows,
        x_labels=list(
            CLASS_NAMES
        ),
        left_label=left_label,
        right_label=right_label,
    )
    axis.set_title(
        title
    )
    axis.legend()

    save_figure(
        figure,
        output_path,
    )
    plt.close(
        figure
    )


def _plot_entropy_rows_on_axis(
    axis,
    rows: list[ChannelEntropyStatisticRow],
    x_labels: list[str],
    left_label: str,
    right_label: str,
) -> None:
    x_positions = np.arange(
        len(rows)
    )
    offset = 0.12

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

    axis.errorbar(
        x_positions - offset,
        left_means,
        yerr=left_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=left_label,
    )
    axis.errorbar(
        x_positions + offset,
        right_means,
        yerr=right_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=right_label,
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    for index, row in enumerate(
        rows
    ):
        label = _significance_label(
            row
        )

        if not label:
            continue

        axis.text(
            index,
            0.98,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xticks(
        x_positions
    )
    axis.set_xticklabels(
        x_labels,
        rotation=20,
        ha="right",
    )
    axis.set_ylabel(
        "Normalized channel entropy"
    )
    axis.grid(
        alpha=0.25,
    )


def _significance_label(
    row: ChannelEntropyStatisticRow,
) -> str:
    if not np.isfinite(
        row.p_value_fdr
    ):
        return ""

    if row.p_value_fdr < 0.001:
        return f"***\nq={row.p_value_fdr:.2g}"

    if row.p_value_fdr < 0.01:
        return f"**\nq={row.p_value_fdr:.2g}"

    if row.p_value_fdr < 0.05:
        return f"*\nq={row.p_value_fdr:.2g}"

    return f"q={row.p_value_fdr:.2g}"
