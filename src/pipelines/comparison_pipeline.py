import numpy as np

from src.data.dataset import get_data_for_subject
from src.pipelines.csp_lda_pipeline import (
    evaluate_csp_lda_for_subject,
    evaluate_csp_lda_time_window_for_subject,
)
from src.pipelines.eegnet_pipeline import (
    evaluate_eegnet_for_subject,
)


SubjectEvaluation = tuple[
    np.ndarray,  # True labels
    float,       # CSP+LDA accuracy
    np.ndarray,  # CSP+LDA predictions
    float,       # EEGNet accuracy
    np.ndarray,  # EEGNet predictions
]

SubjectPredictions = tuple[
    np.ndarray,  # True labels
    np.ndarray,  # CSP+LDA predictions
    np.ndarray,  # EEGNet predictions
]

SubjectAccuracies = tuple[
    float,  # CSP+LDA accuracy
    float,  # EEGNet accuracy
]


def evaluate_models_for_subject(
    subject: int,
) -> SubjectEvaluation | None:
    """Evaluate both models for one subject."""
    data = get_data_for_subject(subject)

    if data is None:
        return None

    _, _, _, y_eval = data

    csp_accuracy, csp_predictions = (
        evaluate_csp_lda_for_subject(
            subject,
            data,
        )
    )

    eegnet_accuracy, eegnet_predictions = (
        evaluate_eegnet_for_subject(
            subject,
            data,
        )
    )

    return (
        np.asarray(y_eval),
        csp_accuracy,
        np.asarray(csp_predictions),
        eegnet_accuracy,
        np.asarray(eegnet_predictions),
    )


def evaluate_models_for_subject_with_different_time_windows_for_csplda(
    subject: int,
    time_windows: list[tuple[float, float]],
) -> tuple[
    np.ndarray,
    dict[
        tuple[float, float],
        tuple[float, np.ndarray],
    ],
    float,
    np.ndarray,
] | None:
    """
    Evaluate CSP+LDA for multiple temporal windows and compare
    them with the original EEGNet model.

    EEGNet is always evaluated using the default epoch window.
    """
    data = get_data_for_subject(subject)

    if data is None:
        return None

    _, _, _, y_eval = data

    eegnet_accuracy, eegnet_predictions = (
        evaluate_eegnet_for_subject(
            subject,
            data,
        )
    )

    csp_results = {}

    for tmin, tmax in time_windows:
        result = (
            evaluate_csp_lda_time_window_for_subject(
                subject,
                tmin,
                tmax,
            )
        )

        if result is None:
            continue

        csp_accuracy, csp_predictions = result

        csp_results[(tmin, tmax)] = (
            csp_accuracy,
            np.asarray(csp_predictions),
        )

    return (
        np.asarray(y_eval),
        csp_results,
        eegnet_accuracy,
        np.asarray(eegnet_predictions),
    )


def get_predictions_for_subject(
    subject: int,
) -> SubjectPredictions | None:
    """Return true labels and predictions from both models."""
    evaluation = evaluate_models_for_subject(subject)

    if evaluation is None:
        return None

    (
        y_true,
        _,
        csp_predictions,
        _,
        eegnet_predictions,
    ) = evaluation

    return (
        y_true,
        csp_predictions,
        eegnet_predictions,
    )


def get_accuracies_for_subject(
    subject: int,
) -> SubjectAccuracies | None:
    """Return accuracies from both models."""
    evaluation = evaluate_models_for_subject(subject)

    if evaluation is None:
        return None

    (
        _,
        csp_accuracy,
        _,
        eegnet_accuracy,
        _,
    ) = evaluation

    return (
        csp_accuracy,
        eegnet_accuracy,
    )


def get_accuracies_for_subject_with_different_time_windows_for_csplda(
    subject: int,
    time_windows: list[tuple[float, float]],
) -> tuple[
    dict[tuple[float, float], float],
    float,
] | None:
    """
    Return CSP+LDA accuracies for multiple temporal windows
    and the original EEGNet accuracy.
    """
    evaluation = (
        evaluate_models_for_subject_with_different_time_windows_for_csplda(
            subject,
            time_windows,
        )
    )

    if evaluation is None:
        return None

    (
        _,
        csp_results,
        eegnet_accuracy,
        _,
    ) = evaluation

    csp_accuracies = {
        time_window: accuracy
        for time_window, (accuracy, _) in csp_results.items()
    }

    return (
        csp_accuracies,
        eegnet_accuracy,
    )


def collect_all_predictions() -> SubjectPredictions:
    """Collect true labels and predictions from all subjects."""
    all_y_true: list[np.ndarray] = []
    all_csp_predictions: list[np.ndarray] = []
    all_eegnet_predictions: list[np.ndarray] = []

    for subject in range(1, 10):
        subject_name = f"A{subject:02d}"

        print(f"\nRunning {subject_name}...")

        predictions = get_predictions_for_subject(subject)

        if predictions is None:
            print(
                f"Skipping {subject_name}: "
                "data not found"
            )
            continue

        (
            y_true,
            csp_predictions,
            eegnet_predictions,
        ) = predictions

        all_y_true.append(y_true)
        all_csp_predictions.append(csp_predictions)
        all_eegnet_predictions.append(eegnet_predictions)

    if not all_y_true:
        raise RuntimeError(
            "No predictions were collected."
        )

    return (
        np.concatenate(all_y_true),
        np.concatenate(all_csp_predictions),
        np.concatenate(all_eegnet_predictions),
    )