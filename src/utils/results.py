import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)
from src.utils.paths import (
    get_results_accuracy_comparison_path,
)


AccuracyResult = tuple[str, float, float]


def save_accuracy_comparison(
    results: list[AccuracyResult],
    output_file: str | Path | None = None,
) -> Path:
    """Save the accuracy comparison to a CSV file."""
    if output_file is None:
        output_file = get_results_accuracy_comparison_path()

    output_file = Path(output_file)

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "subject",
            "csp_lda_accuracy",
            "eegnet_accuracy",
        ])

        writer.writerows(results)

    return output_file


def load_accuracy_comparison(
    input_file: str | Path | None = None,
) -> list[AccuracyResult] | None:
    """Load the accuracy comparison from a CSV file."""
    if input_file is None:
        input_file = get_results_accuracy_comparison_path()

    input_file = Path(input_file)

    if not input_file.exists():
        return None

    results = []

    with input_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append((
                row["subject"],
                float(row["csp_lda_accuracy"]),
                float(row["eegnet_accuracy"]),
            ))

    return results


def print_accuracy_comparison(
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

    csp_accuracies: list[float] = []
    eegnet_accuracies: list[float] = []

    for subject_name, csp_accuracy, eegnet_accuracy in results:
        csp_text = (
            f"{csp_accuracy:.4f} "
            f"({csp_accuracy * 100:.1f}%)"
        )
        eegnet_text = (
            f"{eegnet_accuracy:.4f} "
            f"({eegnet_accuracy * 100:.1f}%)"
        )
        difference = eegnet_accuracy - csp_accuracy

        print(
            f"{subject_name:<10} "
            f"{csp_text:<20} "
            f"{eegnet_text:<20} "
            f"{difference:<20.4f}"
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


def print_classification_report_comparison(
    y_true: np.ndarray,
    csp_predictions: np.ndarray,
    eegnet_predictions: np.ndarray,
) -> None:
    """Print classification reports for both models."""
    print("\n" + "=" * 70)
    print("CSP+LDA — All subjects")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            csp_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("\n" + "=" * 70)
    print("EEGNet — All subjects")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            eegnet_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )