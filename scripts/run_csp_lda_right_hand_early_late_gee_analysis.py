import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.analysis.temporal_statistical_analysis import (
    run_csp_lda_right_hand_early_late_gee_analysis,
)
from src.utils.paths import (
    get_temporal_csp_lda_right_hand_early_late_gee_results_path,
    get_temporal_csp_lda_right_hand_early_late_gee_trials_path,
)


def main() -> None:
    """
    Run the targeted CSP+LDA Right-Hand Early/Late GEE analyses.
    """
    run_csp_lda_right_hand_early_late_gee_analysis()

    print()
    print("Saved CSP+LDA Right-Hand Early/Late GEE analyses:")
    print(
        get_temporal_csp_lda_right_hand_early_late_gee_trials_path()
    )
    print(
        get_temporal_csp_lda_right_hand_early_late_gee_results_path()
    )


if __name__ == "__main__":
    main()
