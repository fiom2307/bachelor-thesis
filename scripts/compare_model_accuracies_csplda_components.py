from collections import defaultdict

import numpy as np

from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject,
    get_accuracies_for_subject_with_different_n_components_for_csplda,
)


NComponentsAccuracyResult = tuple[
    str,    # subject name
    int,    # number of CSP components
    float,  # CSP+LDA accuracy with this number of components
    float,  # CSP+LDA baseline accuracy
    float,  # EEGNet accuracy
]


N_COMPONENTS_LIST = [
    6,
    8,
    10,
    12,
]


def run_experiment() -> list[NComponentsAccuracyResult]:
    """
    Evaluate CSP+LDA with different numbers of CSP components
    for every subject.

    The original CSP+LDA and EEGNet accuracies are also
    included as baselines.
    """
    results = []

    for subject in range(1, 10):
        subject_name = f"A{subject:02d}"

        print(f"\nRunning {subject_name}...")

        component_results = (
            get_accuracies_for_subject_with_different_n_components_for_csplda(
                subject,
                N_COMPONENTS_LIST,
            )
        )

        if component_results is None:
            continue

        csp_component_accuracies, eegnet_accuracy = (
            component_results
        )

        baseline_results = get_accuracies_for_subject(
            subject,
        )

        if baseline_results is None:
            continue

        csp_baseline_accuracy, _ = baseline_results

        for (
            n_components,
            csp_component_accuracy,
        ) in csp_component_accuracies.items():
            results.append(
                (
                    subject_name,
                    n_components,
                    csp_component_accuracy,
                    csp_baseline_accuracy,
                    eegnet_accuracy,
                )
            )

    return results


def print_n_components_accuracy_comparison(
    results: list[NComponentsAccuracyResult],
) -> None:
    """
    Print CSP+LDA accuracies grouped by number of CSP components.

    Each configuration is compared with the original CSP+LDA
    and EEGNet models.
    """
    grouped_results = defaultdict(list)

    for result in results:
        (
            subject_name,
            n_components,
            csp_component_accuracy,
            csp_baseline_accuracy,
            eegnet_accuracy,
        ) = result

        grouped_results[n_components].append(
            (
                subject_name,
                csp_component_accuracy,
                csp_baseline_accuracy,
                eegnet_accuracy,
            )
        )

    for (
        n_components,
        component_results,
    ) in grouped_results.items():
        print()
        print("=" * 90)
        print(
            f"CSP+LDA number of components: "
            f"{n_components}"
        )
        print("=" * 90)

        print(
            f"{'Subject':<12}"
            f"{'CSP+LDA components':<22}"
            f"{'CSP+LDA baseline':<22}"
            f"{'EEGNet':<15}"
            f"{'Δ baseline':<15}"
        )

        print("-" * 90)

        component_accuracies = []
        baseline_accuracies = []
        eegnet_accuracies = []

        for (
            subject_name,
            csp_component_accuracy,
            csp_baseline_accuracy,
            eegnet_accuracy,
        ) in component_results:
            difference_baseline = (
                csp_component_accuracy
                - csp_baseline_accuracy
            )

            component_accuracies.append(
                csp_component_accuracy
            )

            baseline_accuracies.append(
                csp_baseline_accuracy
            )

            eegnet_accuracies.append(
                eegnet_accuracy
            )

            print(
                f"{subject_name:<12}"
                f"{csp_component_accuracy:<22.4f}"
                f"{csp_baseline_accuracy:<22.4f}"
                f"{eegnet_accuracy:<15.4f}"
                f"{difference_baseline:+.4f}"
            )

        mean_components = float(
            np.mean(component_accuracies)
        )

        mean_baseline = float(
            np.mean(baseline_accuracies)
        )

        mean_eegnet = float(
            np.mean(eegnet_accuracies)
        )

        mean_improvement = (
            mean_components
            - mean_baseline
        )

        gap_to_eegnet = (
            mean_eegnet
            - mean_components
        )

        print("-" * 90)

        print(
            f"{'Mean':<12}"
            f"{mean_components:<22.4f}"
            f"{mean_baseline:<22.4f}"
            f"{mean_eegnet:<15.4f}"
            f"{mean_improvement:+.4f}"
        )

        print()

        print(
            f"Mean change vs CSP+LDA baseline: "
            f"{mean_improvement:+.4f} "
            f"({mean_improvement * 100:+.2f} pp)"
        )

        print(
            f"Mean gap to EEGNet: "
            f"{gap_to_eegnet:+.4f} "
            f"({gap_to_eegnet * 100:+.2f} pp)"
        )


def main() -> None:
    results = run_experiment()

    if not results:
        raise RuntimeError(
            "No experiment results were generated."
        )

    print_n_components_accuracy_comparison(
        results,
    )


if __name__ == "__main__":
    main()