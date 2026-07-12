import numpy as np

from src.data.dataset import get_data_for_subject
from src.data.preprocessing import apply_car
from src.pipelines.csp_lda_pipeline import run_csp_lda_for_subject
from src.pipelines.eegnet_pipeline import run_eegnet_for_subject
from src.utils.paths import get_all_confusion_matrices_path
from src.utils.plots import (
    create_confusion_matrix_comparison,
    save_figure,
    show_figure,
)


def collect_all_predictions() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Collect true labels and predictions from all subjects."""
    all_y_true = []
    all_csp_predictions = []
    all_eegnet_predictions = []

    for subject in range(1, 10):
        subject_name = f"A{subject:02d}"

        print(f"\nRunning {subject_name}...")

        data = get_data_for_subject(subject)

        if data is None:
            print(f"Skipping {subject_name}: data not found")
            continue

        X_train, y_train, X_eval, y_eval = data

        # Apply the same CAR preprocessing used during the experiment
        X_train = apply_car(X_train)
        X_eval = apply_car(X_eval)

        data_car = (
            X_train,
            y_train,
            X_eval,
            y_eval,
        )

        csp_accuracy, csp_predictions = (
            run_csp_lda_for_subject(
                subject,
                data_car,
            )
        )

        eegnet_accuracy, eegnet_predictions = (
            run_eegnet_for_subject(
                subject,
                data_car,
            )
        )

        print(
            f"{subject_name}: "
            f"CSP+LDA={csp_accuracy:.4f}, "
            f"EEGNet={eegnet_accuracy:.4f}"
        )

        all_y_true.append(np.asarray(y_eval))
        all_csp_predictions.append(
            np.asarray(csp_predictions)
        )
        all_eegnet_predictions.append(
            np.asarray(eegnet_predictions)
        )

    if not all_y_true:
        raise RuntimeError("No predictions were collected.")

    return (
        np.concatenate(all_y_true),
        np.concatenate(all_csp_predictions),
        np.concatenate(all_eegnet_predictions),
    )


def main():
    y_true, csp_predictions, eegnet_predictions = (
        collect_all_predictions()
    )

    figure = create_confusion_matrix_comparison(
        y_true=y_true,
        csp_predictions=csp_predictions,
        eegnet_predictions=eegnet_predictions,
        subject_name="All subjects",
    )

    output_file = save_figure(
        figure,
        get_all_confusion_matrices_path(),
    )

    print(f"\nConfusion matrices saved to:\n{output_file}")

    show_figure()


if __name__ == "__main__":
    main()