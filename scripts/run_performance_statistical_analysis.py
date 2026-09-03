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
    get_performance_accuracy_statistics_path,
    get_performance_class_recall_profiles_path,
    get_performance_class_recall_statistics_path,
)


def main() -> None:
    """
    Run the performance-level statistical analysis.
    """
    run_performance_statistical_analysis()

    print()
    print("Saved performance statistical analysis:")
    print(
        get_performance_accuracy_statistics_path()
    )
    print(
        get_performance_class_recall_profiles_path()
    )
    print(
        get_performance_class_recall_statistics_path()
    )


if __name__ == "__main__":
    main()
