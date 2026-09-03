import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.temporal_statistical_analysis import (
    run_left_hand_early_relevance_gee_analysis,
)
from src.utils.paths import (
    get_temporal_left_hand_early_gee_results_path,
    get_temporal_left_hand_early_gee_trials_path,
)


def main() -> None:
    """
    Run the targeted Left-Hand early-relevance GEE analysis.
    """
    run_left_hand_early_relevance_gee_analysis()

    print()
    print("Saved Left-Hand early-relevance GEE analysis:")
    print(
        get_temporal_left_hand_early_gee_trials_path()
    )
    print(
        get_temporal_left_hand_early_gee_results_path()
    )


if __name__ == "__main__":
    main()
