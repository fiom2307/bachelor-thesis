import csv

from src.utils.paths import get_results_accuracy_comparison_path


def save_accuracy_comparison(results, output_file=get_results_accuracy_comparison_path()):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "subject",
            "csp_lda_accuracy",
            "eegnet_accuracy"
        ])

        for subject_name, csp_acc, eegnet_acc in results:

            writer.writerow([
                subject_name,
                csp_acc,
                eegnet_acc
            ])

    return output_file


def load_accuracy_comparison(input_file=get_results_accuracy_comparison_path()):
    if not input_file.exists():
        return None

    results = []

    with open(input_file, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject_name = row["subject"]
            csp_acc = float(row["csp_lda_accuracy"])
            eegnet_acc = float(row["eegnet_accuracy"])

            results.append((subject_name, csp_acc, eegnet_acc))

    return results