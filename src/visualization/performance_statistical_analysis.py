import numpy as np
import matplotlib.pyplot as plt

from src.analysis.performance_statistical_analysis import (
    LEFT_LABEL,
    RIGHT_LABEL,
    PerformanceStatisticRow,
)
from src.utils.paths import (
    get_performance_classwise_plot_path,
    get_performance_overall_plot_path,
)
from src.visualization.common import save_figure


def plot_performance_statistical_summaries(
    accuracy_row: PerformanceStatisticRow,
    recall_rows: list[PerformanceStatisticRow],
) -> list:
    """
    Plot overall.png and classwise.png for performance statistics.
    """
    output_paths = []

    overall_path = get_performance_overall_plot_path()
    _plot_overall_accuracy(
        row=accuracy_row,
        output_path=overall_path,
    )
    output_paths.append(
        overall_path
    )

    classwise_path = get_performance_classwise_plot_path()
    _plot_classwise_recall(
        rows=recall_rows,
        output_path=classwise_path,
    )
    output_paths.append(
        classwise_path
    )

    return output_paths


def _plot_overall_accuracy(
    row: PerformanceStatisticRow,
    output_path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(5, 5),
        constrained_layout=True,
    )

    means = np.asarray(
        [
            row.mean_a,
            row.mean_b,
        ],
        dtype=np.float64,
    )
    stds = np.asarray(
        [
            row.std_a,
            row.std_b,
        ],
        dtype=np.float64,
    )
    x_positions = np.arange(
        len(means)
    )

    axis.errorbar(
        x_positions,
        means,
        yerr=stds,
        marker="o",
        capsize=3,
        linewidth=2,
        linestyle="",
    )

    y_max = _finite_y_max(
        means,
        stds,
    )

    axis.set_ylim(
        0.0,
        min(
            1.0,
            y_max * 1.2,
        ),
    )

    if np.isfinite(
        row.p_value
    ):
        axis.text(
            0.5,
            min(
                0.98,
                y_max * 1.08,
            ),
            f"p={row.p_value:.2g}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xticks(
        x_positions
    )
    axis.set_xticklabels([
        LEFT_LABEL,
        RIGHT_LABEL,
    ])
    axis.set_ylabel(
        "Accuracy"
    )
    axis.set_title(
        "CSP+LDA vs EEGNet"
    )
    axis.grid(
        alpha=0.25,
    )

    save_figure(
        figure,
        output_path,
    )

    plt.close(
        figure
    )


def _plot_classwise_recall(
    rows: list[PerformanceStatisticRow],
    output_path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    class_labels = [
        row.class_name
        for row in rows
    ]
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
        label=LEFT_LABEL,
    )

    axis.errorbar(
        x_positions + offset,
        right_means,
        yerr=right_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label=RIGHT_LABEL,
    )

    y_max = _finite_y_max(
        np.concatenate([
            left_means,
            right_means,
        ]),
        np.concatenate([
            left_stds,
            right_stds,
        ]),
    )

    axis.set_ylim(
        0.0,
        min(
            1.0,
            y_max * 1.28,
        ),
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
            min(
                0.98,
                y_max * 1.13,
            ),
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
        class_labels
    )
    axis.set_xlabel(
        "Class"
    )
    axis.set_ylabel(
        "Recall"
    )
    axis.set_title(
        "CSP+LDA vs EEGNet class-wise recall"
    )
    axis.legend()
    axis.grid(
        alpha=0.25,
    )

    save_figure(
        figure,
        output_path,
    )

    plt.close(
        figure
    )


def _finite_y_max(
    means: np.ndarray,
    stds: np.ndarray,
) -> float:
    upper = (
        means
        + np.nan_to_num(
            stds,
            nan=0.0,
        )
    )

    if np.any(
        np.isfinite(
            upper
        )
    ):
        return max(
            float(
                np.nanmax(
                    upper
                )
            ),
            float(
                np.finfo(float).eps
            ),
        )

    return 1.0


def _significance_label(
    row: PerformanceStatisticRow,
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
