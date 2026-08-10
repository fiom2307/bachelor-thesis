import numpy as np


FrequencyBand = tuple[float, float]
FrequencyRanking = list[tuple[FrequencyBand, float]]


def compute_frequency_shap_relevance(
    shap_values: np.ndarray,
    labels: np.ndarray,
    trial_mask: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute class-wise mean absolute SHAP relevance
    for selected trials and frequency bands.
    """
    return {
        int(class_id): np.abs(
            shap_values[
                (labels == class_id)
                & trial_mask
            ]
        ).mean(axis=0)
        for class_id in np.unique(labels)
    }


def rank_shap_frequencies(
    frequency_relevance: dict[int, np.ndarray],
    frequency_bands: tuple[FrequencyBand, ...],
) -> dict[int, FrequencyRanking]:
    """
    Rank frequency bands by SHAP relevance for each class.
    """
    rankings = {}

    for class_id, relevance in frequency_relevance.items():
        if len(relevance) != len(frequency_bands):
            raise ValueError(
                "The number of relevance values must match "
                "the number of frequency bands."
            )

        sorted_indices = np.argsort(
            relevance
        )[::-1]

        rankings[class_id] = [
            (
                frequency_bands[index],
                float(relevance[index]),
            )
            for index in sorted_indices
        ]

    return rankings