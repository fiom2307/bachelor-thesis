from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject,
)
from src.utils.results import (
    load_accuracy_comparison,
    save_accuracy_comparison,
)


AccuracyResult = tuple[str, float, float]


def print_results(
    results: list[AccuracyResult],
) -> None:
    """Print the accuracy comparison."""
    if not results:
        print("No accuracy results available.")
        return

    print("-" * 70)
    print(
        f"{'Subject':<10} "
        f"{'CSP+LDA':<20} "
        f"{'EEGNet':<20} "
        f"{'Difference':<20}"
    )
    print("-" * 70)

    csp_accuracies = []
    eegnet_accuracies = []

    for subject_name, csp_accuracy, eegnet_accuracy in results:
        csp_text = (
            f"{csp_accuracy:.4f} "
            f"({csp_accuracy * 100:.1f}%)"
        )
        eegnet_text = (
            f"{eegnet_accuracy:.4f} "
            f"({eegnet_accuracy * 100:.1f}%)"
        )
        difference_text = (
            f"{eegnet_accuracy - csp_accuracy:.4f}"
        )

        print(
            f"{subject_name:<10} "
            f"{csp_text:<20} "
            f"{eegnet_text:<20} "
            f"{difference_text:<20}"
        )

        csp_accuracies.append(csp_accuracy)
        eegnet_accuracies.append(eegnet_accuracy)

    mean_csp = sum(csp_accuracies) / len(csp_accuracies)
    mean_eegnet = (
        sum(eegnet_accuracies) / len(eegnet_accuracies)
    )
    mean_difference = mean_eegnet - mean_csp

    mean_csp_text = (
        f"{mean_csp:.4f} "
        f"({mean_csp * 100:.1f}%)"
    )
    mean_eegnet_text = (
        f"{mean_eegnet:.4f} "
        f"({mean_eegnet * 100:.1f}%)"
    )

    print("-" * 70)
    print(
        f"{'Mean':<10} "
        f"{mean_csp_text:<20} "
        f"{mean_eegnet_text:<20} "
        f"{mean_difference:<20.4f}"
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
        print_results(results)
        return

    results = run_experiment()

    if not results:
        raise RuntimeError(
            "No experiment results were generated."
        )

    save_accuracy_comparison(results)

    print_results(results)


if __name__ == "__main__":
    main()