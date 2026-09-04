import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.experiments.csp_lda_early_weighted import (
    run_csp_lda_early_weighted_experiment,
)
from src.utils.paths import (
    get_csp_lda_early_weighted_accuracy_statistics_path,
    get_csp_lda_early_weighted_class_summary_path,
    get_csp_lda_early_weighted_classification_report_path,
    get_csp_lda_early_weighted_confusion_matrix_path,
    get_csp_lda_early_weighted_selected_lambdas_path,
    get_csp_lda_early_weighted_subject_results_path,
)


def main() -> None:
    """
    Run the Early-weighted CSP+LDA experiment.
    """
    run_csp_lda_early_weighted_experiment()

    print()
    print("Saved Early-weighted CSP+LDA experiment results:")
    print(
        get_csp_lda_early_weighted_subject_results_path()
    )
    print(
        get_csp_lda_early_weighted_selected_lambdas_path()
    )
    print(
        get_csp_lda_early_weighted_class_summary_path()
    )
    print(
        get_csp_lda_early_weighted_classification_report_path()
    )
    print(
        get_csp_lda_early_weighted_confusion_matrix_path()
    )
    print(
        get_csp_lda_early_weighted_accuracy_statistics_path()
    )


if __name__ == "__main__":
    main()
