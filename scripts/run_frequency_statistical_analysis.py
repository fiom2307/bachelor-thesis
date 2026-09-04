import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.frequency_statistical_analysis import (
    COMPARISONS,
    run_frequency_statistical_analysis,
)
from src.utils.paths import (
    get_frequency_statistical_classwise_results_path,
    get_frequency_statistical_overall_results_path,
)
from src.visualization.frequency_statistical_analysis import (
    plot_frequency_statistical_summaries,
)


def main() -> None:
    """
    Run the frequency relevance statistical analysis.
    """
    _, rows_by_output = run_frequency_statistical_analysis()

    plot_paths = plot_frequency_statistical_summaries(
        rows_by_output
    )

    print()
    print("Saved statistical frequency analysis:")

    for comparison in COMPARISONS:
        print(
            get_frequency_statistical_overall_results_path(
                comparison.slug
            )
        )
        print(
            get_frequency_statistical_classwise_results_path(
                comparison.slug
            )
        )

    for plot_path in plot_paths:
        print(
            plot_path
        )


if __name__ == "__main__":
    main()
