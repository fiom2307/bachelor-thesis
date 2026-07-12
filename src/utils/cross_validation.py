from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.utils.config import N_FOLDS


def get_stratified_folds(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """
    Generate stratified training and validation indices for each fold.

    Stratification keeps approximately the same class distribution in
    every fold. Shuffling and a fixed random seed make the splits
    reproducible.

    Yields:
        The fold number, training indices, and validation indices.
    """
    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=seed,
    )

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):
        yield fold, train_idx, val_idx


def average_fold_probabilities(
    probabilities: list[np.ndarray],
) -> np.ndarray:
    """
    Combine the predictions of all fold models using soft voting.

    The class probabilities predicted by the fold models are averaged
    for each trial. The class with the highest average probability is
    selected as the final prediction.
    """
    if not probabilities:
        raise ValueError("No fold probabilities were provided.")

    mean_probabilities = np.mean(
        probabilities,
        axis=0,
    )

    return np.argmax(
        mean_probabilities,
        axis=1,
    )