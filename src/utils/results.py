import csv
from pathlib import Path

from src.utils.paths import get_results_accuracy_comparison_path


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