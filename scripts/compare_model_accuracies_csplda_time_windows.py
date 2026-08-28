from collections import defaultdict

import numpy as np

from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject,
    get_accuracies_for_subject_with_different_time_windows_for_csplda,
)
from src.visualization.csplda_time_window_accuracies import (
    plot_csplda_time_window_accuracy_comparison,
)
from src.utils.paths import (
    get_csp_lda_time_window_accuracy_plot_path,
)


TimeWindowAccuracyResult = tuple[
    str,    # subject name
    float,  # tmin
    float,  # tmax
    float,  # CSP+LDA time-window accuracy
    float,  # CSP+LDA baseline accuracy
    float,  # EEGNet accuracy
]


TIME_WINDOWS = [
    (0.5, 1.5),
    (0.5, 2.0),
    (0.5, 2.5),
    (0.5, 3.0),
    (0.5, 3.5),
    (1.5, 2.5),
    (2.5, 4.0),
]


def run_experiment() -> list[TimeWindowAccuracyResult]:
    """
    Evaluate CSP+LDA with different temporal windows
    for every subject.

    The original CSP+LDA and EEGNet accuracies are also
    included as baselines.
    """
    results = []

    for subject in range(1, 10):
        subject_name = f"A{subject:02d}"

        window_results = (
            get_accuracies_for_subject_with_different_time_windows_for_csplda(
                subject,
                TIME_WINDOWS,
            )
        )

        if window_results is None:
            continue

        csp_window_accuracies, eegnet_accuracy = (
            window_results
        )

        baseline_results = get_accuracies_for_subject(
            subject,
        )

        if baseline_results is None:
            continue

        csp_baseline_accuracy, _ = baseline_results

        for (
            tmin,
            tmax,
        ), csp_window_accuracy in csp_window_accuracies.items():
            results.append(
                (
                    subject_name,
                    tmin,
                    tmax,
                    csp_window_accuracy,
                    csp_baseline_accuracy,
                    eegnet_accuracy,
                )
            )

    return results


def compute_mean_accuracies(
    results: list[TimeWindowAccuracyResult],
) -> tuple[
    list[tuple[float, float]],
    list[float],
    float,
    float,
]:
    """
    Compute mean accuracies across subjects for each temporal window.
    """
    grouped_window_accuracies = defaultdict(list)
    baseline_accuracies_by_subject = {}
    eegnet_accuracies_by_subject = {}

    for (
        subject_name,
        tmin,
        tmax,
        csp_window_accuracy,
        csp_baseline_accuracy,
        eegnet_accuracy,
    ) in results:
        grouped_window_accuracies[
            (tmin, tmax)
        ].append(
            csp_window_accuracy
        )

        baseline_accuracies_by_subject[
            subject_name
        ] = csp_baseline_accuracy

        eegnet_accuracies_by_subject[
            subject_name
        ] = eegnet_accuracy

    ordered_time_windows = [
        window
        for window in TIME_WINDOWS
        if window in grouped_window_accuracies
    ]

    mean_window_accuracies = [
        float(
            np.mean(
                grouped_window_accuracies[window]
            )
        )
        for window in ordered_time_windows
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
        ordered_time_windows,
        mean_window_accuracies,
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
        time_windows,
        mean_window_accuracies,
        mean_baseline_accuracy,
        mean_eegnet_accuracy,
    ) = compute_mean_accuracies(
        results,
    )

    plot_csplda_time_window_accuracy_comparison(
        time_windows=time_windows,
        mean_window_accuracies=mean_window_accuracies,
        mean_baseline_accuracy=mean_baseline_accuracy,
        mean_eegnet_accuracy=mean_eegnet_accuracy,
        output_path=(
            get_csp_lda_time_window_accuracy_plot_path()
        ),
    )


if __name__ == "__main__":
    main()