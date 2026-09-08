from __future__ import annotations

from pathlib import Path

from src.analysis.channel_statistical_analysis import (
    COMPARISONS,
    ChannelStatisticRow,
    build_subject_roi_profiles,
    collect_subject_normalized_channel_profiles,
    compute_channel_statistics,
    print_channel_statistics,
    _classwise_output_key,
    _overall_output_key,
    _save_channel_statistics_rows,
)
from src.utils.paths import (
    get_channel_sensorimotor_statistical_classwise_results_path,
    get_channel_sensorimotor_statistical_overall_results_path,
)


SENSORIMOTOR_ROIS: tuple[
    tuple[str, tuple[str, ...]],
    ...,
] = (
    (
        "Left sensorimotor",
        (
            "FC3",
            "FC1",
            "C5",
            "C3",
            "C1",
            "CP3",
            "CP1",
        ),
    ),
    (
        "Midline sensorimotor",
        (
            "FCz",
            "Cz",
            "CPz",
        ),
    ),
    (
        "Right sensorimotor",
        (
            "FC2",
            "FC4",
            "C2",
            "C4",
            "C6",
            "CP2",
            "CP4",
        ),
    ),
)


def run_channel_sensorimotor_statistical_analysis() -> dict[
    str,
    list[ChannelStatisticRow],
]:
    """
    Run subject-level sensorimotor-channel relevance statistics.
    """
    normalized_profiles = collect_subject_normalized_channel_profiles()
    profiles = build_subject_roi_profiles(
        normalized_profiles,
        SENSORIMOTOR_ROIS,
    )
    rows_by_output = compute_channel_statistics(
        profiles,
        spatial_rois=SENSORIMOTOR_ROIS,
    )

    save_channel_sensorimotor_statistics(
        rows_by_output
    )
    print_channel_statistics(
        rows_by_output
    )

    return rows_by_output


def save_channel_sensorimotor_statistics(
    rows_by_output: dict[str, list[ChannelStatisticRow]],
) -> list[Path]:
    """
    Save sensorimotor overall_statistics.csv and classwise_statistics.csv.
    """
    output_paths = []

    for comparison in COMPARISONS:
        overall_path = (
            get_channel_sensorimotor_statistical_overall_results_path(
                comparison.slug
            )
        )
        _save_channel_statistics_rows(
            rows=rows_by_output[
                _overall_output_key(
                    comparison
                )
            ],
            output_file=overall_path,
        )
        output_paths.append(
            overall_path
        )

        classwise_path = (
            get_channel_sensorimotor_statistical_classwise_results_path(
                comparison.slug
            )
        )
        _save_channel_statistics_rows(
            rows=rows_by_output[
                _classwise_output_key(
                    comparison
                )
            ],
            output_file=classwise_path,
        )
        output_paths.append(
            classwise_path
        )

    return output_paths
