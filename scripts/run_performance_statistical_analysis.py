import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.performance_statistical_analysis import (
    run_performance_statistical_analysis,
)
from src.utils.paths import (
    get_performance_classwise_results_path,
    get_performance_overall_results_path,
)
from src.visualization.performance_statistical_analysis import (
    plot_performance_statistical_summaries,
)


def main() -> None:
    """
    Run the performance-level statistical analysis.
    """
    accuracy_row, _, recall_rows = (
        run_performance_statistical_analysis()
    )

    plot_paths = plot_performance_statistical_summaries(
        accuracy_row=accuracy_row,
        recall_rows=recall_rows,
    )

    print()
    print("Saved performance statistical analysis:")
    print(
        get_performance_overall_results_path()
    )
    print(
        get_performance_classwise_results_path()
    )

    for plot_path in plot_paths:
        print(
            plot_path
        )


if __name__ == "__main__":
    main()
