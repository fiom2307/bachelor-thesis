from collections import defaultdict

import numpy as np

from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject,
    get_accuracies_for_subject_with_different_time_windows_for_csplda,
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

        print(f"\nRunning {subject_name}...")

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


def print_time_window_accuracy_comparison(
    results: list[TimeWindowAccuracyResult],
) -> None:
    """
    Print CSP+LDA accuracies grouped by temporal window.

    Each temporal-window CSP+LDA model is compared with
    the original CSP+LDA and EEGNet models.
    """
    grouped_results = defaultdict(list)

    for result in results:
        (
            subject_name,
            tmin,
            tmax,
            csp_window_accuracy,
            csp_baseline_accuracy,
            eegnet_accuracy,
        ) = result

        grouped_results[(tmin, tmax)].append(
            (
                subject_name,
                csp_window_accuracy,
                csp_baseline_accuracy,
                eegnet_accuracy,
            )
        )

    for (tmin, tmax), window_results in grouped_results.items():
        print()
        print("=" * 90)
        print(
            f"CSP+LDA time window: "
            f"{tmin:.1f}-{tmax:.1f} s"
        )
        print("=" * 90)

        print(
            f"{'Subject':<12}"
            f"{'CSP+LDA window':<20}"
            f"{'CSP+LDA baseline':<22}"
            f"{'EEGNet':<15}"
            f"{'Δ baseline':<15}"
        )

        print("-" * 90)

        window_accuracies = []
        baseline_accuracies = []
        eegnet_accuracies = []

        for (
            subject_name,
            csp_window_accuracy,
            csp_baseline_accuracy,
            eegnet_accuracy,
        ) in window_results:
            difference_baseline = (
                csp_window_accuracy
                - csp_baseline_accuracy
            )

            window_accuracies.append(
                csp_window_accuracy
            )

            baseline_accuracies.append(
                csp_baseline_accuracy
            )

            eegnet_accuracies.append(
                eegnet_accuracy
            )

            print(
                f"{subject_name:<12}"
                f"{csp_window_accuracy:<20.4f}"
                f"{csp_baseline_accuracy:<22.4f}"
                f"{eegnet_accuracy:<15.4f}"
                f"{difference_baseline:+.4f}"
            )

        mean_window = float(
            np.mean(window_accuracies)
        )

        mean_baseline = float(
            np.mean(baseline_accuracies)
        )

        mean_eegnet = float(
            np.mean(eegnet_accuracies)
        )

        mean_improvement = (
            mean_window
            - mean_baseline
        )

        gap_to_eegnet = (
            mean_eegnet
            - mean_window
        )

        print("-" * 90)

        print(
            f"{'Mean':<12}"
            f"{mean_window:<20.4f}"
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

    print_time_window_accuracy_comparison(
        results,
    )


if __name__ == "__main__":
    main()