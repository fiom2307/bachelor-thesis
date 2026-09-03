import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.experiments.csp_lda_temporal_segmented import (
    run_temporal_segmented_csp_lda_experiment,
)
from src.utils.paths import (
    get_csp_lda_temporal_segmented_class_summary_path,
    get_csp_lda_temporal_segmented_classification_report_path,
    get_csp_lda_temporal_segmented_confusion_matrix_path,
    get_csp_lda_temporal_segmented_subject_results_path,
)


def main() -> None:
    """
    Run the temporal-segmented CSP+LDA feature experiment.
    """
    run_temporal_segmented_csp_lda_experiment()

    print()
    print("Saved temporal-segmented CSP+LDA experiment results:")
    print(
        get_csp_lda_temporal_segmented_subject_results_path()
    )
    print(
        get_csp_lda_temporal_segmented_class_summary_path()
    )
    print(
        get_csp_lda_temporal_segmented_classification_report_path()
    )
    print(
        get_csp_lda_temporal_segmented_confusion_matrix_path()
    )


if __name__ == "__main__":
    main()
