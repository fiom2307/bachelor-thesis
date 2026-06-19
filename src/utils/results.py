import csv

from src.utils.paths import ACCURACY_COMPARISON_FILE


def save_accuracy_comparison(results, output_file=ACCURACY_COMPARISON_FILE):
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "subject",
            "csp_lda_accuracy",
            "eegnet_accuracy",
            "difference_eegnet_minus_csp_lda",
        ])

        for subject_name, csp_acc, eegnet_acc in results:
            difference = eegnet_acc - csp_acc

            writer.writerow([
                subject_name,
                csp_acc,
                eegnet_acc,
                difference,
            ])

    return output_file


def load_accuracy_comparison(input_file=ACCURACY_COMPARISON_FILE):
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