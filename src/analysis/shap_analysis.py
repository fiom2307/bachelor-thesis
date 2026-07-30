from dataclasses import dataclass

import numpy as np
import shap
from tensorflow import keras

from src.utils.config import BASE_SEED


@dataclass(frozen=True)
class SHAPResult:
    """
    SHAP results for one subject.
    """

    values: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray

    @property
    def correct_mask(self) -> np.ndarray:
        """Return correctly classified trials."""
        return self.predictions == self.labels


def select_shap_background(
    data: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 40,
    seed: int = BASE_SEED,
) -> np.ndarray:
    """
    Select an approximately class-balanced SHAP background.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    samples_per_class, remainder = divmod(
        n_samples,
        len(classes),
    )

    selected_indices = []

    for index, class_id in enumerate(classes):
        class_indices = np.flatnonzero(
            labels == class_id
        )

        class_samples = (
            samples_per_class
            + int(index < remainder)
        )

        selected_indices.extend(
            rng.choice(
                class_indices,
                size=class_samples,
                replace=False,
            )
        )

    rng.shuffle(selected_indices)

    return data[
        np.asarray(selected_indices)
    ]


def compute_eegnet_ensemble_shap(
    models: list[keras.Model],
    background_sets: list[np.ndarray],
    data: np.ndarray,
    labels: np.ndarray,
    nsamples: int = 200,
    batch_size: int = 16,
    seed: int = BASE_SEED,
) -> SHAPResult:
    """
    Compute and average SHAP values across EEGNet fold models.
    """
    fold_shap_values = []
    fold_probabilities = []

    for fold_index, (
        model,
        background_data,
    ) in enumerate(
        zip(
            models,
            background_sets,
            strict=True,
        )
    ):
        shap_values, probabilities = _compute_model_shap(
            model=model,
            background_data=background_data,
            data=data,
            labels=labels,
            nsamples=nsamples,
            batch_size=batch_size,
            seed=seed + fold_index,
        )

        fold_shap_values.append(
            shap_values
        )

        fold_probabilities.append(
            probabilities
        )

    mean_shap_values = np.mean(
        fold_shap_values,
        axis=0,
    )

    mean_probabilities = np.mean(
        fold_probabilities,
        axis=0,
    )

    return SHAPResult(
        values=mean_shap_values,
        probabilities=mean_probabilities,
        predictions=mean_probabilities.argmax(axis=1),
        labels=labels,
    )


def compute_class_shap_relevance(
    result: SHAPResult,
    class_labels: list[int],
    correct_only: bool = True,
) -> dict[int, np.ndarray]:
    """
    Compute mean absolute SHAP relevance by class.

    Returns one channel-time matrix per class.
    """
    relevance = {}

    for class_id in class_labels:
        trial_mask = (
            result.labels == class_id
        )

        if correct_only:
            trial_mask &= result.correct_mask

        if not np.any(trial_mask):
            continue

        relevance[class_id] = np.mean(
            np.abs(result.values[trial_mask]),
            axis=0,
        )

    return relevance


def compute_temporal_shap_relevance(
    class_relevance: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """
    Average class-wise SHAP relevance across channels.
    """
    return {
        class_id: relevance.mean(axis=0)
        for class_id, relevance
        in class_relevance.items()
    }


def compute_topographic_shap_relevance(
    class_relevance: dict[int, np.ndarray],
    times: np.ndarray,
    imagery_window: tuple[float, float] = (0.5, 3.5),
) -> dict[int, np.ndarray]:
    """
    Average class-wise SHAP relevance across an imagery interval.
    """
    window_start, window_end = imagery_window

    time_mask = (
        (times >= window_start)
        & (times <= window_end)
    )

    return {
        class_id: relevance[
            :,
            time_mask,
        ].mean(axis=1)
        for class_id, relevance
        in class_relevance.items()
    }


def count_shap_trials_by_class(
    result: SHAPResult,
    class_labels: list[int],
    correct_only: bool = True,
) -> dict[int, int]:
    """
    Count trials included in the class-wise SHAP analysis.
    """
    counts = {}

    for class_id in class_labels:
        trial_mask = (
            result.labels == class_id
        )

        if correct_only:
            trial_mask &= result.correct_mask

        counts[class_id] = int(
            trial_mask.sum()
        )

    return counts


def _compute_model_shap(
    model: keras.Model,
    background_data: np.ndarray,
    data: np.ndarray,
    labels: np.ndarray,
    nsamples: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute true-class SHAP values for one EEGNet model.
    """
    background_data = np.asarray(
        background_data,
        dtype=np.float32,
    )

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    model_background, added_axis = _prepare_model_input(
        model,
        background_data,
    )

    model_data, _ = _prepare_model_input(
        model,
        data,
    )

    probabilities = model.predict(
        model_data,
        batch_size=batch_size,
        verbose=0,
    )

    shap_values = np.empty_like(
        model_data,
        dtype=np.float32,
    )

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(
            labels == class_id
        )

        class_output = model.output[:, int(class_id)]

        explainer = shap.GradientExplainer(
            (
                model.inputs[0],
                class_output,
            ),
            model_background,
        )

        for start in range(
            0,
            len(class_indices),
            batch_size,
        ):
            batch_indices = class_indices[
                start : start + batch_size
            ]

            batch_data = model_data[
                batch_indices
            ]

            batch_shap_values = explainer.shap_values(
                batch_data,
                nsamples=nsamples,
                rseed=(
                    seed
                    + int(class_id) * 1000
                    + start
                ),
            )

            shap_values[batch_indices] = (
                _unwrap_shap_values(
                    batch_shap_values,
                    expected_shape=batch_data.shape,
                )
            )

    if added_axis is not None:
        shap_values = np.squeeze(
            shap_values,
            axis=added_axis,
        )

    return (
        shap_values,
        np.asarray(probabilities),
    )


def _prepare_model_input(
    model: keras.Model,
    data: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    """
    Add EEGNet's singleton input dimension when necessary.
    """
    expected_shape = tuple(
        model.input_shape[1:]
    )

    if data.shape[1:] == expected_shape:
        return data, None

    for axis in range(
        1,
        data.ndim + 1,
    ):
        candidate = np.expand_dims(
            data,
            axis=axis,
        )

        candidate_shape = candidate.shape[1:]

        matches = all(
            expected is None
            or actual == expected
            for actual, expected in zip(
                candidate_shape,
                expected_shape,
                strict=True,
            )
        )

        if matches:
            return candidate, axis

    raise ValueError(
        f"Cannot adapt data shape {data.shape} "
        f"to model input shape {model.input_shape}."
    )


def _unwrap_shap_values(
    values: object,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Remove SHAP's additional single-output dimension."""
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.shape == (*expected_shape, 1):
        values = values[..., 0]

    return values