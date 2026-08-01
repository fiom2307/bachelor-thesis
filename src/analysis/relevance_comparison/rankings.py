from collections.abc import Mapping, Sequence

import numpy as np

from src.data.labels import CLASS_LABELS


ChannelRelevance = Mapping[
    int,
    np.ndarray,
]

ChannelRanking = list[
    tuple[str, float]
]

ChannelRankings = Mapping[
    int,
    ChannelRanking,
]

ChannelFrequency = dict[
    int,
    dict[str, int],
]

NormalizedChannelFrequency = dict[
    int,
    dict[str, float],
]


def rank_group_channels(
    mean_channel_relevance: ChannelRelevance,
    channel_names: list[str],
) -> dict[int, ChannelRanking]:
    """
    Rank channels using mean relevance across subjects.
    """
    _validate_channel_names(
        channel_names
    )

    rankings = {}

    for class_id in CLASS_LABELS:
        if class_id not in mean_channel_relevance:
            continue

        relevance = np.asarray(
            mean_channel_relevance[class_id],
            dtype=np.float64,
        )

        if relevance.ndim != 1:
            raise ValueError(
                "Channel relevance must be one-dimensional."
            )

        if len(relevance) != len(
            channel_names
        ):
            raise ValueError(
                "The number of channel relevance values must "
                "match the number of channel names."
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


def count_top_channel_frequency(
    subject_rankings: Sequence[ChannelRankings],
    channel_names: list[str],
    top_n: int = 5,
) -> ChannelFrequency:
    """
    Count how often each channel appears among the top-ranked channels.
    """
    _validate_frequency_inputs(
        subject_rankings=subject_rankings,
        channel_names=channel_names,
        top_n=top_n,
    )

    frequency = {
        class_id: {
            channel_name: 0
            for channel_name in channel_names
        }
        for class_id in CLASS_LABELS
    }

    valid_channels = set(
        channel_names
    )

    for subject_index, rankings in enumerate(
        subject_rankings,
        start=1,
    ):
        for class_id in CLASS_LABELS:
            if class_id not in rankings:
                continue

            ranking = rankings[
                class_id
            ]

            if len(ranking) < top_n:
                raise ValueError(
                    f"Subject {subject_index}, class {class_id}: "
                    f"ranking contains fewer than {top_n} channels."
                )

            top_channels = [
                channel_name
                for channel_name, _
                in ranking[:top_n]
            ]

            unknown_channels = (
                set(top_channels)
                - valid_channels
            )

            if unknown_channels:
                raise ValueError(
                    f"Unknown channels in ranking: "
                    f"{sorted(unknown_channels)}"
                )

            for channel_name in top_channels:
                frequency[
                    class_id
                ][
                    channel_name
                ] += 1

    return frequency


def normalize_top_channel_frequency(
    channel_frequency: ChannelFrequency,
    subject_rankings: Sequence[ChannelRankings],
) -> NormalizedChannelFrequency:
    """
    Convert top-channel counts to proportions of available subjects.
    """
    normalized_frequency = {}

    for class_id in CLASS_LABELS:
        if class_id not in channel_frequency:
            continue

        subject_count = sum(
            class_id in rankings
            for rankings in subject_rankings
        )

        if subject_count == 0:
            continue

        normalized_frequency[
            class_id
        ] = {
            channel_name: (
                count
                / subject_count
            )
            for channel_name, count
            in channel_frequency[
                class_id
            ].items()
        }

    return normalized_frequency


def rank_channel_frequency(
    channel_frequency: Mapping[
        int,
        Mapping[str, int | float],
    ],
) -> dict[
    int,
    list[tuple[str, float]],
]:
    """
    Rank channels by their top-channel frequency.
    """
    rankings = {}

    for class_id in CLASS_LABELS:
        if class_id not in channel_frequency:
            continue

        class_frequency = channel_frequency[
            class_id
        ]

        rankings[class_id] = sorted(
            (
                (
                    channel_name,
                    float(frequency),
                )
                for channel_name, frequency
                in class_frequency.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

    return rankings


def _validate_frequency_inputs(
    subject_rankings: Sequence[ChannelRankings],
    channel_names: list[str],
    top_n: int,
) -> None:
    """
    Validate inputs used for top-channel frequency.
    """
    if not subject_rankings:
        raise ValueError(
            "At least one subject ranking is required."
        )

    _validate_channel_names(
        channel_names
    )

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    if top_n > len(channel_names):
        raise ValueError(
            "top_n cannot exceed the number of channels."
        )


def _validate_channel_names(
    channel_names: list[str],
) -> None:
    """
    Validate EEG channel names.
    """
    if not channel_names:
        raise ValueError(
            "At least one channel name is required."
        )

    if len(channel_names) != len(
        set(channel_names)
    ):
        raise ValueError(
            "Channel names must be unique."
        )