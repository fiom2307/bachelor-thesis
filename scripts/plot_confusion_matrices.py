from src.utils.paths import (
    get_all_confusion_matrices_path,
    get_subject_confusion_matrices_path,
)
from src.utils.plots import (
    create_confusion_matrix_comparison,
    save_figure,
)
from src.pipelines.comparison_pipeline import (
    collect_all_predictions,
    get_predictions_for_subject,
)


def create_all_subjects_confusion_matrices() -> None:
    """Create one confusion-matrix figure using all subjects together."""
    output_path = get_all_confusion_matrices_path()

    if output_path.exists():
        print(
            "\nThe all-subject confusion matrices already exist:"
            f"\n{output_path}"
        )
        return

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
        output_path,
    )

    print(
        "\nAll-subject confusion matrices saved to:"
        f"\n{output_file}"
    )


def create_confusion_matrices_per_subject() -> None:
    """Create one confusion-matrix figure for each subject."""
    for subject in range(1, 10):
        subject_name = f"A{subject:02d}"
        output_path = get_subject_confusion_matrices_path(subject)

        if output_path.exists():
            print(
                f"\nConfusion matrices for {subject_name} "
                f"already exist:\n{output_path}"
            )
            continue

        predictions = get_predictions_for_subject(subject)

        if predictions is None:
            continue

        y_true, csp_predictions, eegnet_predictions = predictions

        figure = create_confusion_matrix_comparison(
            y_true=y_true,
            csp_predictions=csp_predictions,
            eegnet_predictions=eegnet_predictions,
            subject_name=subject_name,
        )

        output_file = save_figure(
            figure,
            output_path,
        )

        print(
            f"\nConfusion matrices for {subject_name} saved to:"
            f"\n{output_file}"
        )


def create_confusion_matrices(mode: str) -> None:
    """Create confusion matrices according to the selected mode."""
    if mode == "all":
        create_all_subjects_confusion_matrices()

    elif mode == "per_subject":
        create_confusion_matrices_per_subject()

    elif mode == "both":
        create_all_subjects_confusion_matrices()
        create_confusion_matrices_per_subject()

    else:
        raise ValueError(
            "Invalid mode. Use 'all', 'per_subject', or 'both'."
        )


def main() -> None:
    mode = "both"

    create_confusion_matrices(mode)


if __name__ == "__main__":
    main()