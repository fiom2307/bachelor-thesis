"""Functions for computing ERD/ERS from EEG trials."""

from __future__ import annotations

import numpy as np


def select_trials_by_class(
    X: np.ndarray,
    y: np.ndarray,
    class_id: int,
) -> np.ndarray:
    """
    Select all trials belonging to one motor-imagery class.

    Parameters
    ----------
    X
        EEG trials with shape:
        (n_trials, n_channels, n_samples)

    y
        Integer class labels with shape:
        (n_trials,)

    class_id
        Class to select.

    Returns
    -------
    np.ndarray
        Trials belonging to the requested class.
    """
    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "(n_trials, n_channels, n_samples)."
        )

    if y.ndim != 1:
        raise ValueError("y must contain integer class labels.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of trials.")

    class_trials = X[y == class_id]

    if len(class_trials) == 0:
        raise ValueError(
            f"No trials were found for class {class_id}."
        )

    return class_trials