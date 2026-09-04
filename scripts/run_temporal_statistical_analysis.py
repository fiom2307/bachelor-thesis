import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.temporal_statistical_analysis import (
    COMPARISONS,
    run_temporal_statistical_analysis,
)
from src.utils.paths import (
    get_temporal_statistical_classwise_results_path,
    get_temporal_statistical_overall_results_path,
)
from src.visualization.temporal_statistical_analysis import (
    plot_temporal_statistical_summaries,
)


def main() -> None:
    """
    Run the temporal relevance statistical analysis.
    """
    _, rows_by_output = run_temporal_statistical_analysis()

    plot_paths = plot_temporal_statistical_summaries(
        rows_by_output
    )

    print()
    print("Saved statistical temporal analysis:")

    for comparison in COMPARISONS:
        print(
            get_temporal_statistical_overall_results_path(
                comparison.slug
            )
        )
        print(
            get_temporal_statistical_classwise_results_path(
                comparison.slug
            )
        )

    for plot_path in plot_paths:
        print(
            plot_path
        )


if __name__ == "__main__":
    main()
