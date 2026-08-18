from src.pipelines.comparison_pipeline_with_svm import (
    get_accuracies_for_subject,
)
from src.utils.results import (
    AccuracyResult,
    load_accuracy_comparison_with_svm,
    print_accuracy_comparison,
    save_accuracy_comparison_with_svm,
)


def run_experiment() -> list[AccuracyResult]:
    """Evaluate CSP+SVM and EEGNet for every subject."""
    results = []

    for subject in range(1, 10):
        subject_name = f"A{subject:02d}"

        print(f"\nRunning {subject_name}...")

        accuracies = get_accuracies_for_subject(subject)

        if accuracies is None:
            print(
                f"Skipping {subject_name}: "
                "data not found"
            )
            continue

        csp_accuracy, eegnet_accuracy = accuracies

        results.append(
            (
                subject_name,
                csp_accuracy,
                eegnet_accuracy,
            )
        )

    return results


def main() -> None:
    results = load_accuracy_comparison_with_svm()

    if results is not None:
        print_accuracy_comparison(results, csp_model_name="CSP+SVM",)
        return

    results = run_experiment()

    if not results:
        raise RuntimeError(
            "No experiment results were generated."
        )

    save_accuracy_comparison_with_svm(results)

    print_accuracy_comparison(results, csp_model_name="CSP+SVM",)


if __name__ == "__main__":
    main()