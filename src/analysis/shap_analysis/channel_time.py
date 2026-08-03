from typing import Literal

import numpy as np

from analysis.shap_analysis.shap_analysis import SHAPResult


TrialSelection = Literal[
    "correct",
    "incorrect",
]


def compute_class_shap_relevance(
    result: SHAPResult,
    class_labels: list[int],
    trial_selection: TrialSelection,
) -> dict[int, np.ndarray]:
    """
    Compute mean absolute SHAP relevance by true class.

    Returns one channel-time matrix per class for either correctly
    or incorrectly classified trials.
    """
    relevance = {}

    selection_mask = _get_trial_selection_mask(
        result=result,
        trial_selection=trial_selection,
    )

    for class_id in class_labels:
        trial_mask = (
            (result.labels == class_id)
            & selection_mask
        )

        if not np.any(trial_mask):
            continue

        relevance[class_id] = np.mean(
            np.abs(
                result.values[
                    trial_mask
                ]
            ),
            axis=0,
        )

    return relevance


def count_shap_trials_by_class(
    result: SHAPResult,
    class_labels: list[int],
    trial_selection: TrialSelection,
) -> dict[int, int]:
    """
    Count selected trials for each true class.
    """
    selection_mask = _get_trial_selection_mask(
        result=result,
        trial_selection=trial_selection,
    )

    counts = {}

    for class_id in class_labels:
        trial_mask = (
            (result.labels == class_id)
            & selection_mask
        )

        counts[class_id] = int(
            trial_mask.sum()
        )

    return counts


def _get_trial_selection_mask(
    result: SHAPResult,
    trial_selection: TrialSelection,
) -> np.ndarray:
    """
    Return the mask for correctly or incorrectly classified trials.
    """
    if trial_selection == "correct":
        return result.correct_mask

    if trial_selection == "incorrect":
        return result.incorrect_mask

    raise ValueError(
        f"Unsupported trial selection: {trial_selection}"
    )