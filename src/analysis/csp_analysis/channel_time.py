import numpy as np

from src.analysis.csp_analysis.csp import (
    CSPAnalysisResult,
)


def compute_class_csp_relevance(
    result: CSPAnalysisResult,
    class_labels: list[int],
    correct_only: bool = True,
) -> dict[int, np.ndarray]:
    """
    Compute mean absolute channel-time relevance by class.
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
            np.abs(
                result.values[
                    trial_mask
                ]
            ),
            axis=0,
        )

    return relevance


def count_csp_trials_by_class(
    result: CSPAnalysisResult,
    class_labels: list[int],
    correct_only: bool = True,
) -> dict[int, int]:
    """
    Count trials included in the class-wise CSP analysis.
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