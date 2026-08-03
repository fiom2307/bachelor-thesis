import matplotlib.pyplot as plt

from src.analysis.csp_pattern_analysis import (
    compute_subject_csp_patterns,
    load_csp_pattern_result,
    save_csp_pattern_result,
)
from src.data.dataset import (
    get_data_for_subject,
    get_evaluation_erd_epochs_for_subject,
)
from src.models.csp_lda import (
    train_or_load_csp_lda,
)
from src.utils.paths import (
    get_csp_pattern_values_path,
    get_csp_subject_patterns_path,
)
from src.visualization.common import save_figure
from src.visualization.csp_pattern_plots import (
    plot_csp_patterns,
)


def plot_csp_patterns_for_subject(
    subject: int,
) -> None:
    """
    Compute and save mean CSP spatial patterns for one subject.
    """
    data = get_data_for_subject(
        subject
    )

    if data is None:
        print(
            f"Skipping A{subject:02d}: "
            "data unavailable."
        )
        return

    X_train, y_train, _, _ = data

    pattern_file = get_csp_pattern_values_path(
        subject
    )

    if pattern_file.exists():
        print(
            f"Loading saved CSP patterns: "
            f"{pattern_file}"
        )

        pattern_result = load_csp_pattern_result(
            pattern_file
        )
    else:
        print(
            f"Computing CSP patterns for "
            f"A{subject:02d}..."
        )

        models = train_or_load_csp_lda(
            subject,
            X_train,
            y_train,
        )

        pattern_result = (
            compute_subject_csp_patterns(
                models
            )
        )

        save_csp_pattern_result(
            result=pattern_result,
            output_file=pattern_file,
        )

    epochs, _ = (
        get_evaluation_erd_epochs_for_subject(
            subject
        )
    )

    figure = plot_csp_patterns(
        patterns=pattern_result.mean_patterns,
        info=epochs.info,
        subject=subject,
    )

    save_figure(
        figure,
        get_csp_subject_patterns_path(
            subject
        ),
    )

    plt.close(
        figure
    )

    print(
        f"A{subject:02d}: "
        f"saved mean CSP patterns "
        f"({pattern_result.mean_patterns.shape[0]} "
        f"components)."
    )


def main() -> None:
    """
    Generate CSP spatial-pattern plots for all subjects.
    """
    for subject in range(
        1,
        10,
    ):
        plot_csp_patterns_for_subject(
            subject
        )


if __name__ == "__main__":
    main()