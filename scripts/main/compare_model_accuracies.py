from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject,
)
from src.utils.results import (
    AccuracyResult,
    load_accuracy_comparison,
    print_accuracy_comparison,
    save_accuracy_comparison,
)


def run_experiment() -> list[AccuracyResult]:
    """Evaluate both models for every subject."""
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
    results = load_accuracy_comparison()

    if results is not None:
        print_accuracy_comparison(results)
        return

    results = run_experiment()

    if not results:
        raise RuntimeError(
            "No experiment results were generated."
        )

    save_accuracy_comparison(results)

    print_accuracy_comparison(results)


if __name__ == "__main__":
    main()