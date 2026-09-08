from src.analysis.channel_statistical_analysis import (
    COMPARISONS,
    ChannelStatisticRow,
)
from src.utils.paths import (
    get_channel_sensorimotor_statistical_classwise_plot_path,
    get_channel_sensorimotor_statistical_overall_plot_path,
)
from src.visualization.channel_statistical_analysis import (
    _plot_classwise_comparison,
    _plot_overall_comparison,
)


def plot_channel_sensorimotor_statistical_summaries(
    rows_by_output: dict[str, list[ChannelStatisticRow]],
) -> list:
    """
    Plot overall.png and classwise.png per sensorimotor comparison.
    """
    output_paths = []

    for comparison in COMPARISONS:
        overall_rows = rows_by_output.get(
            f"{comparison.slug}:overall",
            [],
        )

        if overall_rows:
            output_path = (
                get_channel_sensorimotor_statistical_overall_plot_path(
                    comparison.slug
                )
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
                get_channel_sensorimotor_statistical_classwise_plot_path(
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
