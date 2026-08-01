from collections.abc import Mapping, Sequence

import numpy as np

from src.data.labels import CLASS_LABELS


ClassRelevance = Mapping[int, np.ndarray]
GroupRelevance = dict[int, np.ndarray]


def aggregate_channel_relevance(
    subject_relevance: Sequence[ClassRelevance],
    normalize: bool = True,
) -> tuple[
    GroupRelevance,
    GroupRelevance,
]:
    """
    Compute channel-wise mean and standard deviation across subjects.

    Each subject must provide one channel-relevance vector per class.
    """
    return _aggregate_relevance(
        subject_relevance=subject_relevance,
        expected_ndim=1,
        normalize=normalize,
    )


def aggregate_temporal_relevance(
    subject_relevance: Sequence[ClassRelevance],
    normalize: bool = True,
) -> tuple[
    GroupRelevance,
    GroupRelevance,
]:
    """
    Compute temporal mean and standard deviation across subjects.

    Each subject must provide one temporal-relevance vector per class.
    """
    return _aggregate_relevance(
        subject_relevance=subject_relevance,
        expected_ndim=1,
        normalize=normalize,
    )


def count_subjects_by_class(
    subject_relevance: Sequence[ClassRelevance],
) -> dict[int, int]:
    """
    Count subjects with available relevance values for each class.
    """
    return {
        class_id: sum(
            class_id in relevance
            for relevance in subject_relevance
        )
        for class_id in CLASS_LABELS
    }


def _aggregate_relevance(
    subject_relevance: Sequence[ClassRelevance],
    expected_ndim: int,
    normalize: bool,
) -> tuple[
    GroupRelevance,
    GroupRelevance,
]:
    """
    Aggregate equally shaped relevance arrays across subjects.
    """
    if not subject_relevance:
        raise ValueError(
            "At least one subject relevance result is required."
        )

    mean_relevance = {}
    standard_deviation = {}

    for class_id in CLASS_LABELS:
        class_values = []

        expected_shape = None

        for subject_index, relevance in enumerate(
            subject_relevance,
            start=1,
        ):
            if class_id not in relevance:
                continue

            values = np.asarray(
                relevance[class_id],
                dtype=np.float64,
            )

            if values.ndim != expected_ndim:
                raise ValueError(
                    f"Subject {subject_index}, class {class_id}: "
                    f"expected a {expected_ndim}D relevance array, "
                    f"but received shape {values.shape}."
                )

            if not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    f"Subject {subject_index}, class {class_id}: "
                    "relevance contains non-finite values."
                )

            if expected_shape is None:
                expected_shape = values.shape
            elif values.shape != expected_shape:
                raise ValueError(
                    f"Subject {subject_index}, class {class_id}: "
                    f"expected shape {expected_shape}, "
                    f"but received {values.shape}."
                )

            if normalize:
                values = _normalize_relevance(
                    values
                )

            class_values.append(
                values
            )

        if not class_values:
            continue

        stacked_values = np.stack(
            class_values,
            axis=0,
        )

        mean_relevance[class_id] = np.mean(
            stacked_values,
            axis=0,
        )

        if len(class_values) == 1:
            standard_deviation[class_id] = np.zeros_like(
                class_values[0]
            )
        else:
            standard_deviation[class_id] = np.std(
                stacked_values,
                axis=0,
                ddof=1,
            )

    if not mean_relevance:
        raise ValueError(
            "No class relevance data was available."
        )

    return (
        mean_relevance,
        standard_deviation,
    )


def _normalize_relevance(
    relevance: np.ndarray,
) -> np.ndarray:
    """
    Normalize relevance values to sum to one.
    """
    total_relevance = float(
        np.sum(
            np.abs(relevance)
        )
    )

    if total_relevance <= np.finfo(float).eps:
        return np.zeros_like(
            relevance,
            dtype=np.float64,
        )

    return (
        relevance
        / total_relevance
    )