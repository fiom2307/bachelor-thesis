import numpy as np

from src.pipelines.comparison_pipeline import (
    get_accuracies_for_subject_with_fbcsp,
)
from src.utils.paths import (
    get_fbcsp_accuracy_comparison_path,
    get_subject_name,
)
from src.visualization.common import (
    save_figure,
)
from src.visualization.model_accuracies_fbcsp import (
    create_fbcsp_accuracy_comparison,
)


FBCSPAccuracyResult = tuple[
    str,    # subject name
    float,  # CSP+LDA accuracy
    float,  # FBCSP+LDA accuracy
    float,  # EEGNet accuracy
]


SUBJECTS = range(
    1,
    10,
)


def run_experiment() -> list[FBCSPAccuracyResult]:
    """
    Compare baseline CSP+LDA, FBCSP+LDA, and EEGNet
    for every subject.
    """
    results = []

    for subject in SUBJECTS:
        subject_name = get_subject_name(
            subject
        )

        (
            csp_lda_accuracy,
            fbcsp_lda_accuracy,
            eegnet_accuracy,
        ) = get_accuracies_for_subject_with_fbcsp(
            subject
        )

        results.append(
            (
                subject_name,
                csp_lda_accuracy,
                fbcsp_lda_accuracy,
                eegnet_accuracy,
            )
        )

    return results


def print_results(
    results: list[FBCSPAccuracyResult],
) -> None:
    """
    Print subject-wise accuracies and mean performance.
    """
    print()
    print(
        "=" * 86
    )

    print(
        f"{'Subject':<10}"
        f"{'CSP+LDA':<18}"
        f"{'FBCSP+LDA':<18}"
        f"{'EEGNet':<18}"
        f"{'FBCSP - CSP':<18}"
    )

    print(
        "-" * 86
    )

    for (
        subject_name,
        csp_accuracy,
        fbcsp_accuracy,
        eegnet_accuracy,
    ) in results:
        difference = (
            fbcsp_accuracy
            - csp_accuracy
        )

        print(
            f"{subject_name:<10}"
            f"{csp_accuracy:<18.4f}"
            f"{fbcsp_accuracy:<18.4f}"
            f"{eegnet_accuracy:<18.4f}"
            f"{difference:+.4f}"
        )

    csp_mean = np.mean(
        [
            result[1]
            for result in results
        ]
    )

    fbcsp_mean = np.mean(
        [
            result[2]
            for result in results
        ]
    )

    eegnet_mean = np.mean(
        [
            result[3]
            for result in results
        ]
    )

    improvement = (
        fbcsp_mean
        - csp_mean
    )

    print(
        "-" * 86
    )

    print(
        f"{'Mean':<10}"
        f"{csp_mean:<18.4f}"
        f"{fbcsp_mean:<18.4f}"
        f"{eegnet_mean:<18.4f}"
        f"{improvement:+.4f}"
    )

    print(
        "=" * 86
    )

    print()

    print(
        "Mean FBCSP improvement over CSP+LDA: "
        f"{improvement * 100:+.2f} percentage points"
    )

    print()


def create_plot(
    results: list[FBCSPAccuracyResult],
) -> None:
    """
    Create and save the accuracy comparison plot.
    """
    subjects = [
        result[0]
        for result in results
    ]

    csp_lda_accuracies = [
        result[1]
        for result in results
    ]

    fbcsp_lda_accuracies = [
        result[2]
        for result in results
    ]

    eegnet_accuracies = [
        result[3]
        for result in results
    ]

    fig = create_fbcsp_accuracy_comparison(
        subjects,
        csp_lda_accuracies,
        fbcsp_lda_accuracies,
        eegnet_accuracies,
    )

    output_path = (
        get_fbcsp_accuracy_comparison_path()
    )

    save_figure(
        fig,
        output_path,
    )

    print(
        "Saved FBCSP accuracy comparison:"
    )

    print(
        output_path
    )


def main() -> None:
    results = run_experiment()

    print_results(
        results
    )

    create_plot(
        results
    )


if __name__ == "__main__":
    main()