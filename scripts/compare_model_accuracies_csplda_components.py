from collections import defaultdict

import numpy as np

from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject,
    get_accuracies_for_subject_with_different_n_components_for_csplda,
)
from src.utils.paths import (
    get_csp_lda_n_components_accuracy_plot_path,
)
from src.visualization.csplda_n_components_accuracies import (
    plot_csplda_n_components_accuracy_comparison,
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
    28,
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


def compute_mean_accuracies(
    results: list[NComponentsAccuracyResult],
) -> tuple[
    list[int],
    list[float],
    float,
    float,
]:
    """
    Compute mean accuracies across subjects for each number
    of CSP components.
    """
    grouped_component_accuracies = defaultdict(list)
    baseline_accuracies_by_subject = {}
    eegnet_accuracies_by_subject = {}

    for (
        subject_name,
        n_components,
        csp_component_accuracy,
        csp_baseline_accuracy,
        eegnet_accuracy,
    ) in results:
        grouped_component_accuracies[
            n_components
        ].append(
            csp_component_accuracy
        )

        baseline_accuracies_by_subject[
            subject_name
        ] = csp_baseline_accuracy

        eegnet_accuracies_by_subject[
            subject_name
        ] = eegnet_accuracy

    ordered_n_components = [
        n_components
        for n_components in N_COMPONENTS_LIST
        if n_components in grouped_component_accuracies
    ]

    mean_component_accuracies = [
        float(
            np.mean(
                grouped_component_accuracies[
                    n_components
                ]
            )
        )
        for n_components in ordered_n_components
    ]

    mean_baseline_accuracy = float(
        np.mean(
            list(
                baseline_accuracies_by_subject.values()
            )
        )
    )

    mean_eegnet_accuracy = float(
        np.mean(
            list(
                eegnet_accuracies_by_subject.values()
            )
        )
    )

    return (
        ordered_n_components,
        mean_component_accuracies,
        mean_baseline_accuracy,
        mean_eegnet_accuracy,
    )


def main() -> None:
    results = run_experiment()

    if not results:
        raise RuntimeError(
            "No experiment results were generated."
        )

    (
        n_components_list,
        mean_component_accuracies,
        mean_baseline_accuracy,
        mean_eegnet_accuracy,
    ) = compute_mean_accuracies(
        results,
    )

    plot_csplda_n_components_accuracy_comparison(
        n_components_list=n_components_list,
        mean_component_accuracies=mean_component_accuracies,
        mean_baseline_accuracy=mean_baseline_accuracy,
        mean_eegnet_accuracy=mean_eegnet_accuracy,
        output_path=(
            get_csp_lda_n_components_accuracy_plot_path()
        ),
    )


if __name__ == "__main__":
    main()
