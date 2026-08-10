import numpy as np


def compute_topographic_shap_relevance(
    class_relevance: dict[int, np.ndarray],
    times: np.ndarray,
    imagery_window: tuple[float, float] = (0.5, 4.0),
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