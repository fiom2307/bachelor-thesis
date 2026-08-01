import csv
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.csp_analysis import (
    compute_channel_csp_relevance,
    compute_class_csp_relevance,
    compute_temporal_csp_relevance,
    load_csp_analysis_result,
    rank_csp_channels,
)
from src.analysis.relevance_comparison import (
    aggregate_channel_relevance,
    aggregate_temporal_relevance,
    compute_channel_spearman,
    compute_subject_channel_spearman,
    compute_subject_top_channel_overlap,
    compute_top_channel_overlap,
    count_top_channel_frequency,
    normalize_top_channel_frequency,
    rank_group_channels,
)
from src.analysis.shap_analysis import (
    compute_channel_shap_relevance,
    compute_class_shap_relevance,
    compute_temporal_shap_relevance,
    load_shap_result,
    rank_shap_channels,
)
from src.data.dataset import (
    get_evaluation_erd_epochs_for_subject,
)
from src.data.labels import (
    CLASS_LABELS,
    CLASS_NAMES,
)
from src.utils.paths import (
    get_csp_values_path,
    get_group_channel_spearman_csv_path,
    get_group_channel_variability_figure_path,
    get_group_global_channel_rankings_csv_path,
    get_group_mean_channel_relevance_csv_path,
    get_group_mean_channel_relevance_figure_path,
    get_group_mean_temporal_relevance_figure_path,
    get_group_top5_channel_frequency_csv_path,
    get_group_top5_channel_frequency_figure_path,
    get_group_top5_overlap_csv_path,
    get_shap_values_path,
    get_subject_channel_spearman_csv_path,
    get_subject_name,
    get_subject_top5_overlap_csv_path,
)
from src.visualization.common import save_figure
from src.visualization.comparison_relevance_plots import (
    plot_group_channel_variability,
    plot_group_mean_channel_relevance,
    plot_group_mean_temporal_relevance,
    plot_top5_channel_frequency,
)


SUBJECTS = range(1, 10)

EPOCH_TMIN = 0.5
IMAGERY_WINDOW = (0.5, 4.0)
TOP_N = 5


def compare_relevance() -> None:
    """
    Aggregate and compare EEGNet SHAP and CSP+LDA relevance.
    """
    class_labels = list(
        CLASS_LABELS
    )

    subject_shap_channel_relevance = {}
    subject_csp_channel_relevance = {}

    subject_shap_temporal_relevance = {}
    subject_csp_temporal_relevance = {}

    subject_shap_rankings = {}
    subject_csp_rankings = {}

    channel_names = None
    shap_times = None
    csp_times = None

    for subject in SUBJECTS:
        shap_file = get_shap_values_path(
            subject
        )

        csp_file = get_csp_values_path(
            subject
        )

        if not shap_file.exists():
            print(
                f"Skipping {get_subject_name(subject)}: "
                f"missing SHAP values."
            )
            continue

        if not csp_file.exists():
            print(
                f"Skipping {get_subject_name(subject)}: "
                f"missing CSP values."
            )
            continue

        print(
            f"Loading relevance results for "
            f"{get_subject_name(subject)}..."
        )

        shap_result = load_shap_result(
            shap_file
        )

        csp_result = load_csp_analysis_result(
            csp_file
        )

        epochs, _ = (
            get_evaluation_erd_epochs_for_subject(
                subject
            )
        )

        current_channel_names = list(
            epochs.ch_names
        )

        channel_names = _set_or_validate_channel_names(
            current=current_channel_names,
            reference=channel_names,
            subject=subject,
        )

        current_shap_times = (
            np.arange(
                shap_result.values.shape[-1]
            )
            / float(
                epochs.info["sfreq"]
            )
            + EPOCH_TMIN
        )

        current_csp_times = np.asarray(
            csp_result.times,
            dtype=np.float64,
        )

        if (
            csp_result.values.shape[-1]
            != len(current_csp_times)
        ):
            raise ValueError(
                f"{get_subject_name(subject)}: "
                "the CSP temporal dimension does not "
                "match csp_result.times."
            )

        shap_times = _set_or_validate_times(
            current=current_shap_times,
            reference=shap_times,
            subject=subject,
            method="SHAP",
        )

        csp_times = _set_or_validate_times(
            current=current_csp_times,
            reference=csp_times,
            subject=subject,
            method="CSP",
        )

        shap_class_relevance = (
            compute_class_shap_relevance(
                result=shap_result,
                class_labels=class_labels,
                correct_only=True,
            )
        )

        csp_class_relevance = (
            compute_class_csp_relevance(
                result=csp_result,
                class_labels=class_labels,
                correct_only=True,
            )
        )

        shap_channel_relevance = (
            compute_channel_shap_relevance(
                shap_class_relevance
            )
        )

        csp_channel_relevance = (
            compute_channel_csp_relevance(
                csp_class_relevance
            )
        )

        shap_temporal_relevance = (
            compute_temporal_shap_relevance(
                shap_class_relevance
            )
        )

        csp_temporal_relevance = (
            compute_temporal_csp_relevance(
                csp_class_relevance
            )
        )

        shap_rankings = rank_shap_channels(
            channel_relevance=(
                shap_channel_relevance
            ),
            channel_names=channel_names,
        )

        csp_rankings = rank_csp_channels(
            channel_relevance=(
                csp_channel_relevance
            ),
            channel_names=channel_names,
        )

        subject_shap_channel_relevance[
            subject
        ] = shap_channel_relevance

        subject_csp_channel_relevance[
            subject
        ] = csp_channel_relevance

        subject_shap_temporal_relevance[
            subject
        ] = shap_temporal_relevance

        subject_csp_temporal_relevance[
            subject
        ] = csp_temporal_relevance

        subject_shap_rankings[
            subject
        ] = shap_rankings

        subject_csp_rankings[
            subject
        ] = csp_rankings

    if not subject_shap_channel_relevance:
        raise ValueError(
            "No shared SHAP and CSP subject results "
            "were available."
        )

    if (
        channel_names is None
        or shap_times is None
        or csp_times is None
    ):
        raise ValueError(
            "The shared channel or temporal information "
            "could not be determined."
        )

    processed_subjects = sorted(
        subject_shap_channel_relevance
    )

    print(
        "Aggregating subjects: "
        + ", ".join(
            get_subject_name(subject)
            for subject in processed_subjects
        )
    )

    mean_shap_channels, std_shap_channels = (
        aggregate_channel_relevance(
            subject_relevance=list(
                subject_shap_channel_relevance.values()
            ),
            normalize=True,
        )
    )

    mean_csp_channels, std_csp_channels = (
        aggregate_channel_relevance(
            subject_relevance=list(
                subject_csp_channel_relevance.values()
            ),
            normalize=True,
        )
    )

    mean_shap_temporal, std_shap_temporal = (
        aggregate_temporal_relevance(
            subject_relevance=list(
                subject_shap_temporal_relevance.values()
            ),
            normalize=True,
        )
    )

    mean_csp_temporal, std_csp_temporal = (
        aggregate_temporal_relevance(
            subject_relevance=list(
                subject_csp_temporal_relevance.values()
            ),
            normalize=True,
        )
    )

    global_shap_rankings = rank_group_channels(
        mean_channel_relevance=(
            mean_shap_channels
        ),
        channel_names=channel_names,
    )

    global_csp_rankings = rank_group_channels(
        mean_channel_relevance=(
            mean_csp_channels
        ),
        channel_names=channel_names,
    )

    shap_ranking_list = list(
        subject_shap_rankings.values()
    )

    csp_ranking_list = list(
        subject_csp_rankings.values()
    )

    shap_top5_counts = (
        count_top_channel_frequency(
            subject_rankings=shap_ranking_list,
            channel_names=channel_names,
            top_n=TOP_N,
        )
    )

    csp_top5_counts = (
        count_top_channel_frequency(
            subject_rankings=csp_ranking_list,
            channel_names=channel_names,
            top_n=TOP_N,
        )
    )

    shap_top5_frequency = (
        normalize_top_channel_frequency(
            channel_frequency=(
                shap_top5_counts
            ),
            subject_rankings=(
                shap_ranking_list
            ),
        )
    )

    csp_top5_frequency = (
        normalize_top_channel_frequency(
            channel_frequency=(
                csp_top5_counts
            ),
            subject_rankings=(
                csp_ranking_list
            ),
        )
    )

    subject_spearman = (
        compute_subject_channel_spearman(
            first_subject_relevance=(
                subject_shap_channel_relevance
            ),
            second_subject_relevance=(
                subject_csp_channel_relevance
            ),
        )
    )

    group_spearman = compute_channel_spearman(
        first_relevance=mean_shap_channels,
        second_relevance=mean_csp_channels,
    )

    subject_top5_overlap = (
        compute_subject_top_channel_overlap(
            first_subject_rankings=(
                subject_shap_rankings
            ),
            second_subject_rankings=(
                subject_csp_rankings
            ),
            top_n=TOP_N,
        )
    )

    group_top5_overlap = (
        compute_top_channel_overlap(
            first_rankings=(
                global_shap_rankings
            ),
            second_rankings=(
                global_csp_rankings
            ),
            top_n=TOP_N,
        )
    )

    _save_group_method_results(
        method="shap",
        mean_channel_relevance=(
            mean_shap_channels
        ),
        channel_standard_deviation=(
            std_shap_channels
        ),
        mean_temporal_relevance=(
            mean_shap_temporal
        ),
        temporal_standard_deviation=(
            std_shap_temporal
        ),
        times=shap_times,
        global_rankings=(
            global_shap_rankings
        ),
        top5_counts=shap_top5_counts,
        top5_frequency=(
            shap_top5_frequency
        ),
        subject_rankings=(
            subject_shap_rankings
        ),
        channel_names=channel_names,
    )

    _save_group_method_results(
        method="csp",
        mean_channel_relevance=(
            mean_csp_channels
        ),
        channel_standard_deviation=(
            std_csp_channels
        ),
        mean_temporal_relevance=(
            mean_csp_temporal
        ),
        temporal_standard_deviation=(
            std_csp_temporal
        ),
        times=csp_times,
        global_rankings=(
            global_csp_rankings
        ),
        top5_counts=csp_top5_counts,
        top5_frequency=(
            csp_top5_frequency
        ),
        subject_rankings=(
            subject_csp_rankings
        ),
        channel_names=channel_names,
    )

    _save_subject_spearman_csv(
        results=subject_spearman,
        output_file=(
            get_subject_channel_spearman_csv_path()
        ),
    )

    _save_group_spearman_csv(
        results=group_spearman,
        output_file=(
            get_group_channel_spearman_csv_path()
        ),
    )

    _save_subject_overlap_csv(
        results=subject_top5_overlap,
        output_file=(
            get_subject_top5_overlap_csv_path()
        ),
    )

    _save_group_overlap_csv(
        results=group_top5_overlap,
        output_file=(
            get_group_top5_overlap_csv_path()
        ),
    )

    print(
        "Relevance comparison completed for "
        f"{len(processed_subjects)} subjects."
    )


def _save_group_method_results(
    method: str,
    mean_channel_relevance: Mapping[
        int,
        np.ndarray,
    ],
    channel_standard_deviation: Mapping[
        int,
        np.ndarray,
    ],
    mean_temporal_relevance: Mapping[
        int,
        np.ndarray,
    ],
    temporal_standard_deviation: Mapping[
        int,
        np.ndarray,
    ],
    times: np.ndarray,
    global_rankings: Mapping[
        int,
        list[tuple[str, float]],
    ],
    top5_counts: Mapping[
        int,
        Mapping[str, int],
    ],
    top5_frequency: Mapping[
        int,
        Mapping[str, float],
    ],
    subject_rankings: Mapping[
        int,
        Mapping[
            int,
            list[tuple[str, float]],
        ],
    ],
    channel_names: list[str],
) -> None:
    """
    Save group plots and CSV files for one relevance method.
    """
    mean_channel_figure = (
        plot_group_mean_channel_relevance(
            mean_channel_relevance=(
                mean_channel_relevance
            ),
            channel_names=channel_names,
            method=method,
        )
    )

    variability_figure = (
        plot_group_channel_variability(
            channel_standard_deviation=(
                channel_standard_deviation
            ),
            channel_names=channel_names,
            method=method,
        )
    )

    temporal_figure = (
        plot_group_mean_temporal_relevance(
            mean_temporal_relevance=(
                mean_temporal_relevance
            ),
            temporal_standard_deviation=(
                temporal_standard_deviation
            ),
            times=times,
            method=method,
            imagery_window=IMAGERY_WINDOW,
        )
    )

    frequency_figure = (
        plot_top5_channel_frequency(
            normalized_frequency=(
                top5_frequency
            ),
            method=method,
        )
    )

    save_figure(
        mean_channel_figure,
        get_group_mean_channel_relevance_figure_path(
            method
        ),
    )

    save_figure(
        variability_figure,
        get_group_channel_variability_figure_path(
            method
        ),
    )

    save_figure(
        temporal_figure,
        get_group_mean_temporal_relevance_figure_path(
            method
        ),
    )

    save_figure(
        frequency_figure,
        get_group_top5_channel_frequency_figure_path(
            method
        ),
    )

    plt.close(
        mean_channel_figure
    )

    plt.close(
        variability_figure
    )

    plt.close(
        temporal_figure
    )

    plt.close(
        frequency_figure
    )

    _save_mean_channel_relevance_csv(
        mean_relevance=(
            mean_channel_relevance
        ),
        standard_deviation=(
            channel_standard_deviation
        ),
        channel_names=channel_names,
        output_file=(
            get_group_mean_channel_relevance_csv_path(
                method
            )
        ),
    )

    _save_global_rankings_csv(
        rankings=global_rankings,
        output_file=(
            get_group_global_channel_rankings_csv_path(
                method
            )
        ),
    )

    _save_top5_frequency_csv(
        counts=top5_counts,
        frequencies=top5_frequency,
        subject_rankings=subject_rankings,
        channel_names=channel_names,
        output_file=(
            get_group_top5_channel_frequency_csv_path(
                method
            )
        ),
    )


def _save_mean_channel_relevance_csv(
    mean_relevance: Mapping[
        int,
        np.ndarray,
    ],
    standard_deviation: Mapping[
        int,
        np.ndarray,
    ],
    channel_names: list[str],
    output_file: Path,
) -> None:
    """
    Save mean channel relevance and variability.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_id",
            "class_name",
            "channel",
            "mean_relevance",
            "standard_deviation",
        ])

        for class_id in CLASS_LABELS:
            if class_id not in mean_relevance:
                continue

            for channel_index, channel_name in enumerate(
                channel_names
            ):
                writer.writerow([
                    class_id,
                    _get_class_name(
                        class_id
                    ),
                    channel_name,
                    float(
                        mean_relevance[
                            class_id
                        ][
                            channel_index
                        ]
                    ),
                    float(
                        standard_deviation[
                            class_id
                        ][
                            channel_index
                        ]
                    ),
                ])


def _save_global_rankings_csv(
    rankings: Mapping[
        int,
        list[tuple[str, float]],
    ],
    output_file: Path,
) -> None:
    """
    Save global channel rankings.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_id",
            "class_name",
            "rank",
            "channel",
            "relevance",
        ])

        for class_id in CLASS_LABELS:
            if class_id not in rankings:
                continue

            for rank, (
                channel_name,
                relevance,
            ) in enumerate(
                rankings[class_id],
                start=1,
            ):
                writer.writerow([
                    class_id,
                    _get_class_name(
                        class_id
                    ),
                    rank,
                    channel_name,
                    relevance,
                ])


def _save_top5_frequency_csv(
    counts: Mapping[
        int,
        Mapping[str, int],
    ],
    frequencies: Mapping[
        int,
        Mapping[str, float],
    ],
    subject_rankings: Mapping[
        int,
        Mapping[
            int,
            list[tuple[str, float]],
        ],
    ],
    channel_names: list[str],
    output_file: Path,
) -> None:
    """
    Save top-five channel counts and proportions.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_id",
            "class_name",
            "channel",
            "count",
            "subject_count",
            "proportion",
        ])

        for class_id in CLASS_LABELS:
            if class_id not in frequencies:
                continue

            subject_count = sum(
                class_id in rankings
                for rankings in subject_rankings.values()
            )

            ordered_channels = sorted(
                channel_names,
                key=lambda channel: (
                    frequencies[
                        class_id
                    ][
                        channel
                    ]
                ),
                reverse=True,
            )

            for channel_name in ordered_channels:
                writer.writerow([
                    class_id,
                    _get_class_name(
                        class_id
                    ),
                    channel_name,
                    counts[
                        class_id
                    ][
                        channel_name
                    ],
                    subject_count,
                    frequencies[
                        class_id
                    ][
                        channel_name
                    ],
                ])


def _save_subject_spearman_csv(
    results,
    output_file: Path,
) -> None:
    """
    Save subject-wise SHAP–CSP Spearman correlations.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "subject",
            "class_id",
            "class_name",
            "correlation",
            "p_value",
        ])

        for subject in sorted(
            results
        ):
            for class_id in CLASS_LABELS:
                if class_id not in results[
                    subject
                ]:
                    continue

                similarity = results[
                    subject
                ][
                    class_id
                ]

                writer.writerow([
                    get_subject_name(
                        subject
                    ),
                    class_id,
                    _get_class_name(
                        class_id
                    ),
                    similarity.correlation,
                    similarity.p_value,
                ])


def _save_group_spearman_csv(
    results,
    output_file: Path,
) -> None:
    """
    Save group-level SHAP–CSP Spearman correlations.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_id",
            "class_name",
            "correlation",
            "p_value",
        ])

        for class_id in CLASS_LABELS:
            if class_id not in results:
                continue

            similarity = results[
                class_id
            ]

            writer.writerow([
                class_id,
                _get_class_name(
                    class_id
                ),
                similarity.correlation,
                similarity.p_value,
            ])


def _save_subject_overlap_csv(
    results,
    output_file: Path,
) -> None:
    """
    Save subject-wise SHAP–CSP top-five overlap.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "subject",
            "class_id",
            "class_name",
            "overlap_count",
            "overlap_proportion",
            "shared_channels",
        ])

        for subject in sorted(
            results
        ):
            for class_id in CLASS_LABELS:
                if class_id not in results[
                    subject
                ]:
                    continue

                overlap = results[
                    subject
                ][
                    class_id
                ]

                writer.writerow([
                    get_subject_name(
                        subject
                    ),
                    class_id,
                    _get_class_name(
                        class_id
                    ),
                    overlap.count,
                    overlap.proportion,
                    ";".join(
                        overlap.shared_channels
                    ),
                ])


def _save_group_overlap_csv(
    results,
    output_file: Path,
) -> None:
    """
    Save group-level SHAP–CSP top-five overlap.
    """
    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_id",
            "class_name",
            "overlap_count",
            "overlap_proportion",
            "shared_channels",
        ])

        for class_id in CLASS_LABELS:
            if class_id not in results:
                continue

            overlap = results[
                class_id
            ]

            writer.writerow([
                class_id,
                _get_class_name(
                    class_id
                ),
                overlap.count,
                overlap.proportion,
                ";".join(
                    overlap.shared_channels
                ),
            ])


def _set_or_validate_channel_names(
    current: list[str],
    reference: list[str] | None,
    subject: int,
) -> list[str]:
    """
    Set or validate the common channel order.
    """
    if reference is None:
        return current

    if current != reference:
        raise ValueError(
            f"{get_subject_name(subject)}: "
            "the EEG channel order differs from "
            "the previous subjects."
        )

    return reference


def _set_or_validate_times(
    current: np.ndarray,
    reference: np.ndarray | None,
    subject: int,
    method: str,
) -> np.ndarray:
    """
    Set or validate a common temporal axis.
    """
    current = np.asarray(
        current,
        dtype=np.float64,
    )

    if reference is None:
        return current

    if (
        current.shape != reference.shape
        or not np.allclose(
            current,
            reference,
        )
    ):
        raise ValueError(
            f"{get_subject_name(subject)}: "
            f"the {method} temporal axis differs "
            "from the previous subjects."
        )

    return reference


def _get_class_name(
    class_id: int,
) -> str:
    """
    Return the readable name of one class.
    """
    class_index = list(
        CLASS_LABELS
    ).index(
        class_id
    )

    return CLASS_NAMES[
        class_index
    ]


def main() -> None:
    """
    Generate group relevance results and SHAP–CSP comparisons.
    """
    compare_relevance()


if __name__ == "__main__":
    main()