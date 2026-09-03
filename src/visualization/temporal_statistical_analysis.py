import matplotlib.pyplot as plt
import numpy as np

from src.analysis.temporal_statistical_analysis import (
    COMPARISONS,
    ClassModelTemporalStatisticRow,
    ClassTemporalStatisticRow,
    TemporalStatisticRow,
)
from src.data.labels import CLASS_NAMES
from src.utils.paths import (
    get_temporal_classwise_csp_vs_eegnet_correct_plot_path,
    get_temporal_classwise_statistical_plot_path,
    get_temporal_statistical_plot_path,
)


def plot_temporal_statistical_summaries(
    rows: list[TemporalStatisticRow],
) -> list:
    """
    Plot one statistical temporal relevance summary per comparison.
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

        output_path = get_temporal_statistical_plot_path(
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


def plot_csp_lda_classwise_temporal_statistical_summaries(
    rows: list[ClassTemporalStatisticRow],
) -> list:
    """
    Plot one class-wise CSP+LDA temporal summary per class.
    """
    output_paths = []

    for class_name in [
        class_name.title()
        for class_name in CLASS_NAMES
    ]:
        class_rows = [
            row
            for row in rows
            if row.class_name == class_name
        ]

        if not class_rows:
            continue

        output_path = get_temporal_classwise_statistical_plot_path(
            _slugify_class_name(
                class_name
            )
        )

        _plot_classwise_comparison(
            rows=class_rows,
            title=(
                "CSP+LDA correct vs incorrect - "
                f"{class_name}"
            ),
            output_path=output_path,
        )

        output_paths.append(
            output_path
        )

    return output_paths


def plot_classwise_csp_lda_vs_eegnet_correct_temporal_summaries(
    rows: list[ClassModelTemporalStatisticRow],
) -> list:
    """
    Plot class-wise CSP+LDA correct vs EEGNet correct summaries.
    """
    output_paths = []

    for class_name in [
        class_name.title()
        for class_name in CLASS_NAMES
    ]:
        class_rows = [
            row
            for row in rows
            if row.class_name == class_name
        ]

        if not class_rows:
            continue

        output_path = (
            get_temporal_classwise_csp_vs_eegnet_correct_plot_path(
                _slugify_class_name(
                    class_name
                )
            )
        )

        _plot_classwise_model_comparison(
            rows=class_rows,
            title=(
                "CSP+LDA correct vs EEGNet correct - "
                f"{class_name}"
            ),
            output_path=output_path,
        )

        output_paths.append(
            output_path
        )

    return output_paths


def _plot_comparison(
    rows: list[TemporalStatisticRow],
    title: str,
    left_label: str,
    right_label: str,
    output_path,
) -> None:
    """
    Create one temporal-window relevance comparison plot.
    """
    window_labels = [
        (
            f"{row.temporal_window}\n"
            f"{row.start_time:g}-{row.end_time:g} s"
        )
        for row in rows
    ]

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
        figsize=(10, 6),
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
        window_labels,
    )

    axis.set_xlabel(
        "Temporal window"
    )

    axis.set_ylabel(
        "Mean normalized temporal relevance"
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


def _plot_classwise_comparison(
    rows: list[ClassTemporalStatisticRow],
    title: str,
    output_path,
) -> None:
    """
    Create one class-wise temporal-window comparison plot.
    """
    window_labels = [
        (
            f"{row.temporal_window}\n"
            f"{row.start_time:g}-{row.end_time:g} s"
        )
        for row in rows
    ]

    x_positions = np.arange(
        len(rows)
    )

    correct_means = np.asarray(
        [
            row.correct_mean
            for row in rows
        ],
        dtype=np.float64,
    )
    incorrect_means = np.asarray(
        [
            row.incorrect_mean
            for row in rows
        ],
        dtype=np.float64,
    )
    correct_stds = np.asarray(
        [
            row.correct_std
            for row in rows
        ],
        dtype=np.float64,
    )
    incorrect_stds = np.asarray(
        [
            row.incorrect_std
            for row in rows
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    axis.errorbar(
        x_positions,
        correct_means,
        yerr=correct_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label="Correct",
    )

    axis.errorbar(
        x_positions,
        incorrect_means,
        yerr=incorrect_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label="Incorrect",
    )

    finite_upper = np.concatenate([
        correct_means + np.nan_to_num(
            correct_stds,
            nan=0.0,
        ),
        incorrect_means + np.nan_to_num(
            incorrect_stds,
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
        window_labels
    )
    axis.set_xlabel(
        "Temporal window"
    )
    axis.set_ylabel(
        "Mean normalized temporal relevance"
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


def _plot_classwise_model_comparison(
    rows: list[ClassModelTemporalStatisticRow],
    title: str,
    output_path,
) -> None:
    """
    Create one class-wise CSP+LDA vs EEGNet correct plot.
    """
    window_labels = [
        (
            f"{row.temporal_window}\n"
            f"{row.start_time:g}-{row.end_time:g} s"
        )
        for row in rows
    ]

    x_positions = np.arange(
        len(rows)
    )

    csp_means = np.asarray(
        [
            row.csp_correct_mean
            for row in rows
        ],
        dtype=np.float64,
    )
    eegnet_means = np.asarray(
        [
            row.eegnet_correct_mean
            for row in rows
        ],
        dtype=np.float64,
    )
    csp_stds = np.asarray(
        [
            row.csp_correct_std
            for row in rows
        ],
        dtype=np.float64,
    )
    eegnet_stds = np.asarray(
        [
            row.eegnet_correct_std
            for row in rows
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6),
        constrained_layout=True,
    )

    axis.errorbar(
        x_positions,
        csp_means,
        yerr=csp_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label="CSP+LDA correct",
    )

    axis.errorbar(
        x_positions,
        eegnet_means,
        yerr=eegnet_stds,
        marker="o",
        capsize=3,
        linewidth=2,
        label="EEGNet correct",
    )

    finite_upper = np.concatenate([
        csp_means + np.nan_to_num(
            csp_stds,
            nan=0.0,
        ),
        eegnet_means + np.nan_to_num(
            eegnet_stds,
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
        window_labels
    )
    axis.set_xlabel(
        "Temporal window"
    )
    axis.set_ylabel(
        "Mean normalized temporal relevance"
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


def _slugify_class_name(
    class_name: str,
) -> str:
    return (
        class_name.lower()
        .replace(" ", "_")
    )
