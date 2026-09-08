from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.analysis.channel_statistical_analysis import (
    ALPHA,
    COMPARISONS,
    SUBJECTS,
    ChannelComparison,
    ModelName,
    TrialSelection,
    _direction_label,
    _fdr_correct,
    _format_float,
    _nanmean,
    _nanmedian,
    _nanstd,
    _paired_wilcoxon,
    collect_subject_normalized_channel_profiles,
)
from src.data.labels import CLASS_LABELS, CLASS_NAMES
from src.utils.paths import (
    get_channel_entropy_statistical_classwise_results_path,
    get_channel_entropy_statistical_overall_results_path,
)


@dataclass(frozen=True)
class SubjectChannelEntropyProfile:
    subject: int
    model: ModelName
    condition: TrialSelection
    class_values: np.ndarray

    @property
    def values(self) -> float:
        if np.any(
            ~np.isfinite(
                self.class_values
            )
        ):
            return np.nan

        return float(
            np.mean(
                self.class_values
            )
        )


@dataclass(frozen=True)
class ChannelEntropyStatisticRow:
    comparison: str
    comparison_slug: str
    scope: str
    class_name: str
    metric: str
    n: int
    mean_a: float
    std_a: float
    median_a: float
    mean_b: float
    std_b: float
    median_b: float
    mean_difference: float
    median_difference: float
    wilcoxon_statistic: float
    p_value: float
    p_value_fdr: float
    significant_fdr: bool
    direction: str


def run_channel_entropy_statistical_analysis() -> dict[
    str,
    list[ChannelEntropyStatisticRow],
]:
    """
    Run normalized spatial entropy statistics from channel relevance profiles.
    """
    profiles = collect_subject_channel_entropy_profiles()
    rows_by_output = compute_channel_entropy_statistics(
        profiles
    )

    save_channel_entropy_statistics(
        rows_by_output
    )
    print_channel_entropy_statistics(
        rows_by_output
    )

    return rows_by_output


def collect_subject_channel_entropy_profiles() -> list[
    SubjectChannelEntropyProfile
]:
    """
    Compute class-wise normalized spatial entropy per subject/model/condition.
    """
    normalized_profiles = collect_subject_normalized_channel_profiles()

    return [
        SubjectChannelEntropyProfile(
            subject=profile.subject,
            model=profile.model,
            condition=profile.condition,
            class_values=_class_spatial_entropy(
                profile.class_values
            ),
        )
        for profile in normalized_profiles
    ]


def compute_channel_entropy_statistics(
    profiles: list[SubjectChannelEntropyProfile],
) -> dict[str, list[ChannelEntropyStatisticRow]]:
    """
    Compute overall and class-wise entropy Wilcoxon tests.
    """
    profile_lookup = {
        (
            profile.subject,
            profile.model,
            profile.condition,
        ): profile
        for profile in profiles
    }

    rows_by_output = {}

    for comparison in COMPARISONS:
        rows_by_output[
            _overall_output_key(
                comparison
            )
        ] = _compute_overall_entropy_statistics(
            profile_lookup,
            comparison,
        )

        rows_by_output[
            _classwise_output_key(
                comparison
            )
        ] = _compute_classwise_entropy_statistics(
            profile_lookup,
            comparison,
        )

    return rows_by_output


def save_channel_entropy_statistics(
    rows_by_output: dict[str, list[ChannelEntropyStatisticRow]],
) -> list[Path]:
    """
    Save entropy overall_statistics.csv and classwise_statistics.csv.
    """
    output_paths = []

    for comparison in COMPARISONS:
        overall_path = get_channel_entropy_statistical_overall_results_path(
            comparison.slug
        )
        _save_channel_entropy_statistics_rows(
            rows_by_output[
                _overall_output_key(
                    comparison
                )
            ],
            overall_path,
        )
        output_paths.append(
            overall_path
        )

        classwise_path = get_channel_entropy_statistical_classwise_results_path(
            comparison.slug
        )
        _save_channel_entropy_statistics_rows(
            rows_by_output[
                _classwise_output_key(
                    comparison
                )
            ],
            classwise_path,
        )
        output_paths.append(
            classwise_path
        )

    return output_paths


def print_channel_entropy_statistics(
    rows_by_output: dict[str, list[ChannelEntropyStatisticRow]],
) -> None:
    """
    Print compact entropy summaries grouped by comparison.
    """
    for comparison in COMPARISONS:
        print()
        print("=" * 70)
        print(f"{comparison.name} channel entropy")
        print("=" * 70)
        print(
            f"{'Scope':<12} "
            f"{'Class':<22} "
            f"{comparison.left_label + ' mean+/-SD':<28} "
            f"{comparison.right_label + ' mean+/-SD':<28} "
            f"{'A-B':>8} "
            f"{'p_FDR':>10}"
        )

        for row in rows_by_output[
            _overall_output_key(
                comparison
            )
        ]:
            _print_entropy_row(
                row
            )

        for row in rows_by_output[
            _classwise_output_key(
                comparison
            )
        ]:
            _print_entropy_row(
                row
            )


def _compute_overall_entropy_statistics(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectChannelEntropyProfile,
    ],
    comparison: ChannelComparison,
) -> list[ChannelEntropyStatisticRow]:
    left_values, right_values = _paired_overall_entropy_values(
        profile_lookup,
        comparison,
    )
    row = _compute_entropy_statistic_row(
        comparison=comparison,
        scope="overall",
        class_name="all_classes_balanced",
        left_values=left_values,
        right_values=right_values,
    )
    corrected_p_values, significant = _fdr_correct([
        row.p_value
    ])

    return [
        _replace_fdr(
            row,
            corrected_p_values[0],
            bool(
                significant[0]
            ),
        )
    ]


def _compute_classwise_entropy_statistics(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectChannelEntropyProfile,
    ],
    comparison: ChannelComparison,
) -> list[ChannelEntropyStatisticRow]:
    rows = []
    p_values = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        left_values, right_values = _paired_class_entropy_values(
            profile_lookup=profile_lookup,
            comparison=comparison,
            class_index=class_index,
        )
        row = _compute_entropy_statistic_row(
            comparison=comparison,
            scope="classwise",
            class_name=class_name,
            left_values=left_values,
            right_values=right_values,
        )
        rows.append(
            row
        )
        p_values.append(
            row.p_value
        )

    corrected_p_values, significant = _fdr_correct(
        p_values
    )

    return [
        _replace_fdr(
            row,
            p_value_fdr,
            bool(
                is_significant
            ),
        )
        for row, p_value_fdr, is_significant in zip(
            rows,
            corrected_p_values,
            significant,
            strict=True,
        )
    ]


def _compute_entropy_statistic_row(
    comparison: ChannelComparison,
    scope: str,
    class_name: str,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> ChannelEntropyStatisticRow:
    differences = (
        left_values
        - right_values
    )
    statistic, p_value = _paired_wilcoxon(
        differences
    )
    mean_difference = _nanmean(
        differences
    )

    return ChannelEntropyStatisticRow(
        comparison=comparison.name,
        comparison_slug=comparison.slug,
        scope=scope,
        class_name=class_name,
        metric="normalized_channel_entropy",
        n=len(
            differences
        ),
        mean_a=_nanmean(
            left_values
        ),
        std_a=_nanstd(
            left_values
        ),
        median_a=_nanmedian(
            left_values
        ),
        mean_b=_nanmean(
            right_values
        ),
        std_b=_nanstd(
            right_values
        ),
        median_b=_nanmedian(
            right_values
        ),
        mean_difference=mean_difference,
        median_difference=_nanmedian(
            differences
        ),
        wilcoxon_statistic=statistic,
        p_value=p_value,
        p_value_fdr=np.nan,
        significant_fdr=False,
        direction=_direction_label(
            comparison,
            mean_difference,
        ),
    )


def _paired_overall_entropy_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectChannelEntropyProfile,
    ],
    comparison: ChannelComparison,
) -> tuple[np.ndarray, np.ndarray]:
    left_values = []
    right_values = []

    for subject in SUBJECTS:
        left_profile = profile_lookup.get((
            subject,
            *comparison.left_key,
        ))
        right_profile = profile_lookup.get((
            subject,
            *comparison.right_key,
        ))

        if (
            left_profile is None
            or right_profile is None
        ):
            continue

        left_value = left_profile.values
        right_value = right_profile.values

        if not (
            np.isfinite(
                left_value
            )
            and np.isfinite(
                right_value
            )
        ):
            continue

        left_values.append(
            left_value
        )
        right_values.append(
            right_value
        )

    return (
        np.asarray(
            left_values,
            dtype=np.float64,
        ),
        np.asarray(
            right_values,
            dtype=np.float64,
        ),
    )


def _paired_class_entropy_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectChannelEntropyProfile,
    ],
    comparison: ChannelComparison,
    class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_values = []
    right_values = []

    for subject in SUBJECTS:
        left_profile = profile_lookup.get((
            subject,
            *comparison.left_key,
        ))
        right_profile = profile_lookup.get((
            subject,
            *comparison.right_key,
        ))

        if (
            left_profile is None
            or right_profile is None
        ):
            continue

        left_value = left_profile.class_values[
            class_index
        ]
        right_value = right_profile.class_values[
            class_index
        ]

        if not (
            np.isfinite(
                left_value
            )
            and np.isfinite(
                right_value
            )
        ):
            continue

        left_values.append(
            left_value
        )
        right_values.append(
            right_value
        )

    return (
        np.asarray(
            left_values,
            dtype=np.float64,
        ),
        np.asarray(
            right_values,
            dtype=np.float64,
        ),
    )


def _class_spatial_entropy(
    class_profiles: np.ndarray,
) -> np.ndarray:
    entropies = np.full(
        len(CLASS_LABELS),
        np.nan,
        dtype=np.float64,
    )

    for class_index in range(
        class_profiles.shape[0]
    ):
        entropies[
            class_index
        ] = _normalized_entropy(
            class_profiles[class_index]
        )

    return entropies


def _normalized_entropy(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or len(values) < 2
        or np.any(
            ~np.isfinite(
                values
            )
        )
    ):
        return np.nan

    positive_values = values[
        values > 0
    ]

    if len(
        positive_values
    ) == 0:
        return np.nan

    entropy = -np.sum(
        positive_values
        * np.log(
            positive_values
        )
    )

    return float(
        entropy
        / np.log(
            len(values)
        )
    )


def _replace_fdr(
    row: ChannelEntropyStatisticRow,
    p_value_fdr: float,
    significant_fdr: bool,
) -> ChannelEntropyStatisticRow:
    return ChannelEntropyStatisticRow(
        comparison=row.comparison,
        comparison_slug=row.comparison_slug,
        scope=row.scope,
        class_name=row.class_name,
        metric=row.metric,
        n=row.n,
        mean_a=row.mean_a,
        std_a=row.std_a,
        median_a=row.median_a,
        mean_b=row.mean_b,
        std_b=row.std_b,
        median_b=row.median_b,
        mean_difference=row.mean_difference,
        median_difference=row.median_difference,
        wilcoxon_statistic=row.wilcoxon_statistic,
        p_value=row.p_value,
        p_value_fdr=p_value_fdr,
        significant_fdr=significant_fdr,
        direction=row.direction,
    )


def _save_channel_entropy_statistics_rows(
    rows: list[ChannelEntropyStatisticRow],
    output_file: str | Path,
) -> Path:
    output_file = Path(
        output_file
    )
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "comparison",
                "scope",
                "class",
                "metric",
                "n",
                "mean_a",
                "std_a",
                "median_a",
                "mean_b",
                "std_b",
                "median_b",
                "mean_difference",
                "median_difference",
                "wilcoxon_statistic",
                "p_value",
                "p_value_fdr",
                "significant_fdr",
                "direction",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "comparison": row.comparison,
                "scope": row.scope,
                "class": row.class_name,
                "metric": row.metric,
                "n": row.n,
                "mean_a": _format_float(
                    row.mean_a
                ),
                "std_a": _format_float(
                    row.std_a
                ),
                "median_a": _format_float(
                    row.median_a
                ),
                "mean_b": _format_float(
                    row.mean_b
                ),
                "std_b": _format_float(
                    row.std_b
                ),
                "median_b": _format_float(
                    row.median_b
                ),
                "mean_difference": _format_float(
                    row.mean_difference
                ),
                "median_difference": _format_float(
                    row.median_difference
                ),
                "wilcoxon_statistic": _format_float(
                    row.wilcoxon_statistic
                ),
                "p_value": _format_float(
                    row.p_value
                ),
                "p_value_fdr": _format_float(
                    row.p_value_fdr
                ),
                "significant_fdr": row.significant_fdr,
                "direction": row.direction,
            })

    return output_file


def _print_entropy_row(
    row: ChannelEntropyStatisticRow,
) -> None:
    marker = (
        "*"
        if row.significant_fdr
        else ""
    )

    print(
        f"{row.scope:<12} "
        f"{row.class_name:<22} "
        f"{_mean_sd(row.mean_a, row.std_a):<28} "
        f"{_mean_sd(row.mean_b, row.std_b):<28} "
        f"{row.mean_difference:>8.4f} "
        f"{row.p_value_fdr:>10.4g}"
        f"{marker}"
    )


def _mean_sd(
    mean: float,
    std: float,
) -> str:
    return f"{mean:.4f}+/-{std:.4f}"


def _overall_output_key(
    comparison: ChannelComparison,
) -> str:
    return f"{comparison.slug}:overall"


def _classwise_output_key(
    comparison: ChannelComparison,
) -> str:
    return f"{comparison.slug}:classwise"
