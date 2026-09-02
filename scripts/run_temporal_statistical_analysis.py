import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.temporal_statistical_analysis import (
    run_temporal_statistical_analysis,
)
from src.utils.paths import (
    get_temporal_statistical_profiles_path,
    get_temporal_statistical_results_path,
)
from src.visualization.temporal_statistical_analysis import (
    plot_temporal_statistical_summaries,
)


def main() -> None:
    """
    Run the temporal relevance statistical analysis.
    """
    _, rows = run_temporal_statistical_analysis()

    plot_paths = plot_temporal_statistical_summaries(
        rows
    )

    print()
    print("Saved statistical temporal analysis:")
    print(
        get_temporal_statistical_profiles_path()
    )
    print(
        get_temporal_statistical_results_path()
    )

    for plot_path in plot_paths:
        print(
            plot_path
        )


if __name__ == "__main__":
    main()
