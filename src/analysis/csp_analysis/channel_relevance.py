import numpy as np


ChannelRanking = list[
    tuple[str, float]
]


def compute_channel_csp_relevance(
    class_relevance: dict[
        int,
        np.ndarray,
    ],
) -> dict[int, np.ndarray]:
    """
    Average class-wise CSP relevance across time.
    """
    return {
        class_id: relevance.mean(
            axis=1
        )
        for class_id, relevance
        in class_relevance.items()
    }


def rank_csp_channels(
    channel_relevance: dict[
        int,
        np.ndarray,
    ],
    channel_names: list[str],
) -> dict[int, ChannelRanking]:
    """
    Rank EEG channels by CSP relevance for each class.
    """
    rankings = {}

    for class_id, relevance in (
        channel_relevance.items()
    ):
        if len(relevance) != len(
            channel_names
        ):
            raise ValueError(
                "The number of relevance values "
                "must match the number of "
                "channel names."
            )

        sorted_indices = np.argsort(
            relevance
        )[::-1]

        rankings[class_id] = [
            (
                channel_names[index],
                float(relevance[index]),
            )
            for index in sorted_indices
        ]

    return rankings