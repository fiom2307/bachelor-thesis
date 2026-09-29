import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.pipelines.eegnet_pipeline import (
    evaluate_eegnet_time_window_for_subject,
)
from src.utils.paths import (
    get_eegnet_time_window_accuracy_results_path,
    get_subject_name,
)


TimeWindowAccuracyResult = tuple[
    str,
    float,
    float,
    float,
]


TIME_WINDOWS = [
    (0.5, 1.5),
    (2.5, 4.0),
]


def run_experiment() -> list[TimeWindowAccuracyResult]:
    """
    Train/evaluate EEGNet for the Early and Late temporal windows.
    """
    results = []

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )

        print(f"\nRunning {subject_name}...")

        for tmin, tmax in TIME_WINDOWS:
            result = evaluate_eegnet_time_window_for_subject(
                subject,
                tmin,
                tmax,
            )

            if result is None:
                print(
                    f"Skipping {subject_name} "
                    f"{tmin:.1f}-{tmax:.1f} s: data not found"
                )
                continue

            accuracy, _ = result

            results.append(
                (
                    subject_name,
                    tmin,
                    tmax,
                    accuracy,
                )
            )

    return results


def save_results(
    results: list[TimeWindowAccuracyResult],
) -> Path:
    """
    Save subject-wise EEGNet time-window accuracies.
    """
    output_file = get_eegnet_time_window_accuracy_results_path()
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "subject",
            "tmin",
            "tmax",
            "eegnet_accuracy",
        ])
        writer.writerows(results)

    return output_file


def print_summary(
    results: list[TimeWindowAccuracyResult],
) -> None:
    """
    Print Early and Late EEGNet accuracy per subject and mean.
    """
    grouped = defaultdict(dict)

    for subject_name, tmin, tmax, accuracy in results:
        grouped[subject_name][(tmin, tmax)] = accuracy

    print()
    print(f"{'Subject':<8} | {'EEGNet Early':<12} | {'EEGNet Late':<11}")
    print("-" * 40)

    early_values = []
    late_values = []

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )
        early = grouped[subject_name].get((0.5, 1.5), np.nan)
        late = grouped[subject_name].get((2.5, 4.0), np.nan)

        if np.isfinite(early):
            early_values.append(early)

        if np.isfinite(late):
            late_values.append(late)

        print(
            f"{subject_name:<8} | "
            f"{_format_accuracy(early):<12} | "
            f"{_format_accuracy(late):<11}"
        )

    print(
        f"{'Mean':<8} | "
        f"{_format_accuracy(float(np.mean(early_values))):<12} | "
        f"{_format_accuracy(float(np.mean(late_values))):<11}"
    )


def _format_accuracy(
    value: float,
) -> str:
    if not np.isfinite(value):
        return ""

    return f"{value:.4f}"


def main() -> None:
    results = run_experiment()

    if not results:
        raise RuntimeError(
            "No EEGNet time-window results were generated."
        )

    output_file = save_results(
        results
    )

    print_summary(
        results
    )

    print()
    print("Saved EEGNet time-window accuracies:")
    print(output_file)


if __name__ == "__main__":
    main()
