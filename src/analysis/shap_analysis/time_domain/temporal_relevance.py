import numpy as np


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
