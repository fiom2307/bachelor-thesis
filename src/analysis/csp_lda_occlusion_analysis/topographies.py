import numpy as np


def compute_topographic_csp_relevance(
    class_relevance: dict[
        int,
        np.ndarray,
    ],
    window_times: np.ndarray,
    imagery_window: tuple[
        float,
        float,
    ] = (0.5, 4.0),
) -> dict[int, np.ndarray]:
    """
    Average class-wise CSP relevance across an imagery interval.
    """
    window_start, window_end = (
        imagery_window
    )

    time_mask = (
        (window_times >= window_start)
        & (window_times <= window_end)
    )

    if not np.any(time_mask):
        raise ValueError(
            "No occlusion windows fall inside "
            "the imagery interval."
        )

    return {
        class_id: relevance[
            :,
            time_mask,
        ].mean(axis=1)
        for class_id, relevance
        in class_relevance.items()
    }