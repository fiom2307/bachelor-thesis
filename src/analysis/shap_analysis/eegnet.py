from dataclasses import dataclass

import numpy as np
from tensorflow import keras

from src.analysis.shap_analysis._utils import (
    compute_model_shap,
)
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
        shap_values, probabilities = compute_model_shap(
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