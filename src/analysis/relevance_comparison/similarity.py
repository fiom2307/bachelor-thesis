from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

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

SubjectChannelRelevance = Mapping[
    int,
    ChannelRelevance,
]

SubjectChannelRankings = Mapping[
    int,
    ChannelRankings,
]


@dataclass(frozen=True)
class SpearmanSimilarity:
    """
    Spearman similarity between two channel-relevance vectors.
    """

    correlation: float
    p_value: float


@dataclass(frozen=True)
class TopChannelOverlap:
    """
    Overlap between two top-channel rankings.
    """

    count: int
    proportion: float
    shared_channels: tuple[str, ...]


def compute_channel_spearman(
    first_relevance: ChannelRelevance,
    second_relevance: ChannelRelevance,
) -> dict[int, SpearmanSimilarity]:
    """
    Compute class-wise Spearman correlations between channel relevances.
    """
    similarities = {}

    for class_id in CLASS_LABELS:
        if (
            class_id not in first_relevance
            or class_id not in second_relevance
        ):
            continue

        first_values = np.asarray(
            first_relevance[class_id],
            dtype=np.float64,
        )

        second_values = np.asarray(
            second_relevance[class_id],
            dtype=np.float64,
        )

        _validate_relevance_pair(
            first_values=first_values,
            second_values=second_values,
            class_id=class_id,
        )

        result = spearmanr(
            first_values,
            second_values,
        )

        similarities[class_id] = (
            SpearmanSimilarity(
                correlation=float(
                    result.statistic
                ),
                p_value=float(
                    result.pvalue
                ),
            )
        )

    return similarities


def compute_subject_channel_spearman(
    first_subject_relevance: SubjectChannelRelevance,
    second_subject_relevance: SubjectChannelRelevance,
) -> dict[
    int,
    dict[int, SpearmanSimilarity],
]:
    """
    Compute SHAP–CSP channel correlations for each subject and class.
    """
    shared_subjects = sorted(
        set(first_subject_relevance)
        & set(second_subject_relevance)
    )

    if not shared_subjects:
        raise ValueError(
            "No subjects are shared between the relevance results."
        )

    return {
        subject: compute_channel_spearman(
            first_relevance=(
                first_subject_relevance[
                    subject
                ]
            ),
            second_relevance=(
                second_subject_relevance[
                    subject
                ]
            ),
        )
        for subject in shared_subjects
    }


def compute_top_channel_overlap(
    first_rankings: ChannelRankings,
    second_rankings: ChannelRankings,
    top_n: int = 5,
) -> dict[int, TopChannelOverlap]:
    """
    Compute class-wise overlap between two top-channel rankings.
    """
    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    overlaps = {}

    for class_id in CLASS_LABELS:
        if (
            class_id not in first_rankings
            or class_id not in second_rankings
        ):
            continue

        first_ranking = first_rankings[
            class_id
        ]

        second_ranking = second_rankings[
            class_id
        ]

        if (
            len(first_ranking) < top_n
            or len(second_ranking) < top_n
        ):
            raise ValueError(
                f"Class {class_id}: both rankings must "
                f"contain at least {top_n} channels."
            )

        first_channels = [
            channel_name
            for channel_name, _
            in first_ranking[:top_n]
        ]

        second_channel_set = {
            channel_name
            for channel_name, _
            in second_ranking[:top_n]
        }

        shared_channels = tuple(
            channel_name
            for channel_name in first_channels
            if channel_name in second_channel_set
        )

        overlap_count = len(
            shared_channels
        )

        overlaps[class_id] = (
            TopChannelOverlap(
                count=overlap_count,
                proportion=(
                    overlap_count
                    / top_n
                ),
                shared_channels=shared_channels,
            )
        )

    return overlaps


def compute_subject_top_channel_overlap(
    first_subject_rankings: SubjectChannelRankings,
    second_subject_rankings: SubjectChannelRankings,
    top_n: int = 5,
) -> dict[
    int,
    dict[int, TopChannelOverlap],
]:
    """
    Compute top-channel overlap for each subject and class.
    """
    shared_subjects = sorted(
        set(first_subject_rankings)
        & set(second_subject_rankings)
    )

    if not shared_subjects:
        raise ValueError(
            "No subjects are shared between the ranking results."
        )

    return {
        subject: compute_top_channel_overlap(
            first_rankings=(
                first_subject_rankings[
                    subject
                ]
            ),
            second_rankings=(
                second_subject_rankings[
                    subject
                ]
            ),
            top_n=top_n,
        )
        for subject in shared_subjects
    }


def _validate_relevance_pair(
    first_values: np.ndarray,
    second_values: np.ndarray,
    class_id: int,
) -> None:
    """
    Validate two channel-relevance vectors.
    """
    if (
        first_values.ndim != 1
        or second_values.ndim != 1
    ):
        raise ValueError(
            f"Class {class_id}: channel relevance "
            "must be one-dimensional."
        )

    if first_values.shape != second_values.shape:
        raise ValueError(
            f"Class {class_id}: relevance vectors have "
            f"different shapes: {first_values.shape} "
            f"and {second_values.shape}."
        )

    if not (
        np.all(np.isfinite(first_values))
        and np.all(np.isfinite(second_values))
    ):
        raise ValueError(
            f"Class {class_id}: relevance vectors contain "
            "non-finite values."
        )