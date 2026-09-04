import numpy as np
import matplotlib.pyplot as plt

from src.analysis.frequency_statistical_analysis import (
    COMPARISONS,
    FrequencyStatisticRow,
)
from src.data.labels import CLASS_NAMES
from src.utils.paths import (
    get_frequency_statistical_classwise_plot_path,
    get_frequency_statistical_overall_plot_path,
)
from src.visualization.common import save_figure


def plot_frequency_statistical_summaries(
    rows_by_output: dict[str, list[FrequencyStatisticRow]],
) -> list:
    """
    Plot overall.png and classwise.png per frequency comparison.
    """
    output_paths = []

    for comparison in COMPARISONS:
        overall_rows = rows_by_output.get(
            f"{comparison.slug}:overall",
            [],
        )

        if overall_rows:
            output_path = get_frequency_statistical_overall_plot_path(
                comparison.slug
            )

            _plot_overall_comparison(
                rows=overall_rows,
                title=comparison.name,
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
            output_path = (
                get_frequency_statistical_classwise_plot_path(
                    comparison.slug
                )
            )

            _plot_classwise_comparison(
                rows=classwise_rows,
                title=comparison.name,
                left_label=comparison.left_label,
                right_label=comparison.right_label,
                output_path=output_path,
            )

            output_paths.append(
                output_path
            )

    return output_paths


def _plot_overall_comparison(
    rows: list[FrequencyStatisticRow],
    title: str,
    left_label: str,
    right_label: str,
    output_path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5),
        constrained_layout=True,
    )

    _plot_rows_on_axis(
        axis=axis,
        rows=rows,
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


def _plot_classwise_comparison(
    rows: list[FrequencyStatisticRow],
    title: str,
    left_label: str,
    right_label: str,
    output_path,
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharey=True,
        constrained_layout=True,
    )

    for axis, class_name in zip(
        axes.ravel(),
        CLASS_NAMES,
        strict=True,
    ):
        class_rows = [
            row
            for row in rows
            if row.class_name == class_name
        ]

        _plot_rows_on_axis(
            axis=axis,
            rows=class_rows,
            left_label=left_label,
            right_label=right_label,
        )

        axis.set_title(
            class_name
        )

    handles, labels = axes.ravel()[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncols=2,
    )

    figure.suptitle(
        title,
        y=1.02,
    )

    save_figure(
        figure,
        output_path,
    )

    plt.close(
        figure
    )


def _plot_rows_on_axis(
    axis,
    rows: list[FrequencyStatisticRow],
    left_label: str,
    right_label: str,
) -> None:
    band_labels = [
        row.frequency_band
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

    if np.any(
        np.isfinite(
            finite_upper
        )
    ):
        y_max = float(
            np.nanmax(
                finite_upper
            )
        )
    else:
        y_max = 1.0

    y_max = max(
        y_max,
        float(
            np.finfo(float).eps
        ),
    )

    axis.set_ylim(
        0.0,
        y_max * 1.28,
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
            y_max * 1.13,
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
        band_labels
    )

    axis.set_xlabel(
        "Frequency band"
    )

    axis.set_ylabel(
        "Mean relative frequency relevance"
    )

    axis.grid(
        alpha=0.25,
    )


def _significance_label(
    row: FrequencyStatisticRow,
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
