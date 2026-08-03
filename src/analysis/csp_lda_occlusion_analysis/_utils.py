import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)

from src.data.labels import CLASS_LABELS


def compute_model_occlusion(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    reference: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    sfreq: float,
    tmin: float,
    window_duration: float,
    window_step: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute channel-time occlusion relevance for one CSP+LDA model.
    """
    data = np.asarray(
        data,
        dtype=np.float64,
    )

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    if data.ndim != 3:
        raise ValueError(
            "data must have shape "
            "(trials, channels, times)."
        )

    if reference.shape != data.shape[1:]:
        raise ValueError(
            "reference must have shape "
            "(channels, times)."
        )

    if len(labels) != len(data):
        raise ValueError(
            "The number of labels must match "
            "the number of trials."
        )

    window_bounds, window_times = create_temporal_windows(
        n_times=data.shape[-1],
        sfreq=sfreq,
        tmin=tmin,
        window_duration=window_duration,
        window_step=window_step,
    )

    original_probabilities = predict_model_probabilities(
        csp=csp,
        lda=lda,
        data=data,
    )

    label_columns = _get_label_columns(
        labels
    )

    trial_indices = np.arange(
        len(labels)
    )

    original_true_probabilities = original_probabilities[
        trial_indices,
        label_columns,
    ]

    occlusion_values = np.empty(
        (
            data.shape[0],
            data.shape[1],
            len(window_bounds),
        ),
        dtype=np.float32,
    )

    for channel_index in range(
        data.shape[1]
    ):
        for window_index, (
            window_start,
            window_end,
        ) in enumerate(window_bounds):
            perturbed_data = data.copy()

            perturbed_data[
                :,
                channel_index,
                window_start:window_end,
            ] = reference[
                channel_index,
                window_start:window_end,
            ]

            perturbed_probabilities = (
                predict_model_probabilities(
                    csp=csp,
                    lda=lda,
                    data=perturbed_data,
                )
            )

            perturbed_true_probabilities = (
                perturbed_probabilities[
                    trial_indices,
                    label_columns,
                ]
            )

            occlusion_values[
                :,
                channel_index,
                window_index,
            ] = (
                original_true_probabilities
                - perturbed_true_probabilities
            )

    return (
        occlusion_values,
        original_probabilities,
        window_times,
    )


def predict_model_probabilities(
    csp: CSP,
    lda: LinearDiscriminantAnalysis,
    data: np.ndarray,
) -> np.ndarray:
    """
    Predict aligned class probabilities for one CSP+LDA model.
    """
    features = csp.transform(
        data
    )

    probabilities = lda.predict_proba(
        features
    )

    return align_class_probabilities(
        probabilities=probabilities,
        model_classes=lda.classes_,
    )


def create_temporal_windows(
    n_times: int,
    sfreq: float,
    tmin: float,
    window_duration: float,
    window_step: float,
) -> tuple[
    list[tuple[int, int]],
    np.ndarray,
]:
    """
    Create sliding temporal windows and their center times.
    """
    if n_times <= 0:
        raise ValueError(
            "n_times must be greater than zero."
        )

    if sfreq <= 0:
        raise ValueError(
            "sfreq must be greater than zero."
        )

    if window_duration <= 0:
        raise ValueError(
            "window_duration must be greater than zero."
        )

    if window_step <= 0:
        raise ValueError(
            "window_step must be greater than zero."
        )

    window_size = max(
        int(round(window_duration * sfreq)),
        1,
    )

    step_size = max(
        int(round(window_step * sfreq)),
        1,
    )

    if window_size > n_times:
        raise ValueError(
            "The occlusion window is longer "
            "than the EEG epoch."
        )

    final_start = (
        n_times
        - window_size
    )

    start_indices = list(
        range(
            0,
            final_start + 1,
            step_size,
        )
    )

    if start_indices[-1] != final_start:
        start_indices.append(
            final_start
        )

    window_bounds = [
        (
            start,
            start + window_size,
        )
        for start in start_indices
    ]

    window_times = np.asarray(
        [
            tmin
            + (
                start
                + (window_size - 1) / 2
            )
            / sfreq
            for start in start_indices
        ],
        dtype=float,
    )

    return (
        window_bounds,
        window_times,
    )


def align_class_probabilities(
    probabilities: np.ndarray,
    model_classes: np.ndarray,
) -> np.ndarray:
    """
    Align probability columns to the predefined class order.
    """
    class_labels = np.asarray(
        list(CLASS_LABELS),
        dtype=int,
    )

    model_classes = np.asarray(
        model_classes,
        dtype=int,
    )

    aligned_probabilities = np.zeros(
        (
            probabilities.shape[0],
            len(class_labels),
        ),
        dtype=probabilities.dtype,
    )

    for target_index, class_id in enumerate(
        class_labels
    ):
        matches = np.flatnonzero(
            model_classes == class_id
        )

        if len(matches) != 1:
            raise ValueError(
                f"Class {class_id} is missing "
                "from the LDA model."
            )

        aligned_probabilities[
            :,
            target_index,
        ] = probabilities[
            :,
            matches[0],
        ]

    return aligned_probabilities


def _get_label_columns(
    labels: np.ndarray,
) -> np.ndarray:
    """
    Convert class labels to probability-column indices.
    """
    label_to_column = {
        int(class_id): index
        for index, class_id
        in enumerate(CLASS_LABELS)
    }

    try:
        return np.asarray(
            [
                label_to_column[int(label)]
                for label in labels
            ],
            dtype=int,
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown class label: {error.args[0]}"
        ) from error