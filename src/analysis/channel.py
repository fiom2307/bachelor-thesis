from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.analysis.channel_statistical_analysis import (
    ALPHA,
    CHANNEL_NAMES,
    COMPARISONS,
    SUBJECTS,
    ChannelComparison,
    ModelName,
    TrialSelection,
    collect_subject_normalized_channel_profiles,
    _format_float,
    _nanmean,
    _nanmedian,
    _nanstd,
)
from src.utils.paths import (
    get_channel_statistical_profiles_path,
    get_channel_statistical_results_path,
    get_subject_name,
)


@dataclass(frozen=True)
class SubjectChannelProfile:
    subject: int
    model: ModelName
    condition: TrialSelection
    values: np.ndarray


@dataclass(frozen=True)
class ChannelStatisticRow:
    comparison: str
    comparison_slug: str
    channel: str
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


def run_channel_analysis() -> tuple[
    list[SubjectChannelProfile],
    list[ChannelStatisticRow],
]:
    """
    Run the legacy subject-level channel relevance statistical analysis.

    This writes the flat per-channel CSV files used before the ROI-based
    channel statistical analysis was introduced.
    """
    profiles = collect_subject_channel_profiles()
    rows = compute_channel_statistics(
        profiles
    )

    save_channel_profiles(
        profiles
    )
    save_channel_statistics(
        rows
    )
    print_channel_statistics(
        rows
    )

    return profiles, rows


def collect_subject_channel_profiles() -> list[SubjectChannelProfile]:
    """
    Build one class-balanced relative channel profile per subject/model/condition.
    """
    return [
        SubjectChannelProfile(
            subject=profile.subject,
            model=profile.model,
            condition=profile.condition,
            values=profile.values,
        )
        for profile in collect_subject_normalized_channel_profiles()
    ]


def compute_channel_statistics(
    profiles: list[SubjectChannelProfile],
) -> list[ChannelStatisticRow]:
    """
    Compute paired Wilcoxon tests and descriptives per EEG channel.
    """
    profile_lookup = {
        (
            profile.subject,
            profile.model,
            profile.condition,
        ): profile.values
        for profile in profiles
    }

    rows = []

    for comparison in COMPARISONS:
        comparison_rows = []
        raw_p_values = []

        for channel_index, channel_name in enumerate(
            CHANNEL_NAMES
        ):
            left_values, right_values = _paired_channel_values(
                profile_lookup=profile_lookup,
                comparison=comparison,
                channel_index=channel_index,
            )

            row = _compute_statistic_row(
                comparison=comparison,
                channel=channel_name,
                left_values=left_values,
                right_values=right_values,
            )

            comparison_rows.append(
                row
            )
            raw_p_values.append(
                row.p_value
            )

        rows.extend(
            _with_fdr_correction(
                comparison_rows,
                raw_p_values,
            )
        )

    return rows


def save_channel_profiles(
    profiles: list[SubjectChannelProfile],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save subject-level class-balanced relative channel profiles.
    """
    if output_file is None:
        output_file = get_channel_statistical_profiles_path()

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
        writer = csv.writer(
            file
        )

        writer.writerow([
            "subject",
            "model",
            "condition",
            "channel",
            "relative_relevance",
        ])

        for profile in profiles:
            subject_name = get_subject_name(
                profile.subject
            )

            for channel_name, value in zip(
                CHANNEL_NAMES,
                profile.values,
                strict=True,
            ):
                writer.writerow([
                    subject_name,
                    profile.model,
                    profile.condition,
                    channel_name,
                    _format_float(value),
                ])

    return output_file


def save_channel_statistics(
    rows: list[ChannelStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save legacy per-channel statistical test results.
    """
    if output_file is None:
        output_file = get_channel_statistical_results_path()

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
                "channel",
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
                "channel": row.channel,
                "n": row.n,
                "mean_a": _format_float(row.mean_a),
                "std_a": _format_float(row.std_a),
                "median_a": _format_float(row.median_a),
                "mean_b": _format_float(row.mean_b),
                "std_b": _format_float(row.std_b),
                "median_b": _format_float(row.median_b),
                "mean_difference": _format_float(
                    row.mean_difference
                ),
                "median_difference": _format_float(
                    row.median_difference
                ),
                "wilcoxon_statistic": _format_float(
                    row.wilcoxon_statistic
                ),
                "p_value": _format_float(row.p_value),
                "p_value_fdr": _format_float(
                    row.p_value_fdr
                ),
                "significant_fdr": row.significant_fdr,
                "direction": row.direction,
            })

    return output_file


def print_channel_statistics(
    rows: list[ChannelStatisticRow],
) -> None:
    """
    Print compact console summaries grouped by comparison.
    """
    for comparison in COMPARISONS:
        print()
        print("=" * 70)
        print(
            f"{comparison.name} - CHANNEL RELEVANCE"
        )
        print("=" * 70)
        print(
            f"{'Channel':<8} "
            f"{comparison.left_label + ' mean+/-SD':<28} "
            f"{comparison.right_label + ' mean+/-SD':<28} "
            f"{'A-B':>8} "
            f"{'p':>10} "
            f"{'p_FDR':>10}"
        )

        for row in rows:
            if row.comparison_slug != comparison.slug:
                continue

            marker = (
                "*"
                if row.significant_fdr
                else ""
            )

            print(
                f"{row.channel:<8} "
                f"{_mean_sd(row.mean_a, row.std_a):<28} "
                f"{_mean_sd(row.mean_b, row.std_b):<28} "
                f"{row.mean_difference:>8.4f} "
                f"{row.p_value:>10.4g} "
                f"{row.p_value_fdr:>10.4g}"
                f"{marker}"
            )


def _compute_statistic_row(
    comparison: ChannelComparison,
    channel: str,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> ChannelStatisticRow:
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

    return ChannelStatisticRow(
        comparison=comparison.name,
        comparison_slug=comparison.slug,
        channel=channel,
        n=len(differences),
        mean_a=_nanmean(left_values),
        std_a=_nanstd(left_values),
        median_a=_nanmedian(left_values),
        mean_b=_nanmean(right_values),
        std_b=_nanstd(right_values),
        median_b=_nanmedian(right_values),
        mean_difference=mean_difference,
        median_difference=_nanmedian(
            differences
        ),
        wilcoxon_statistic=statistic,
        p_value=p_value,
        p_value_fdr=np.nan,
        significant_fdr=False,
        direction=_direction_label(
            comparison=comparison,
            mean_difference=mean_difference,
        ),
    )


def _paired_channel_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        np.ndarray,
    ],
    comparison: ChannelComparison,
    channel_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return paired left/right values for one comparison and channel.
    """
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

        left_value = left_profile[
            channel_index
        ]
        right_value = right_profile[
            channel_index
        ]

        if not (
            np.isfinite(left_value)
            and np.isfinite(right_value)
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


def _paired_wilcoxon(
    differences: np.ndarray,
) -> tuple[float, float]:
    """
    Compute a two-sided paired Wilcoxon signed-rank test.
    """
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )
    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if len(differences) < 2:
        return np.nan, np.nan

    if np.allclose(
        differences,
        0.0,
    ):
        return 0.0, 1.0

    statistic, p_value = wilcoxon(
        differences,
        alternative="two-sided",
    )

    return (
        float(statistic),
        float(p_value),
    )


def _with_fdr_correction(
    rows: list[ChannelStatisticRow],
    p_values: list[float],
) -> list[ChannelStatisticRow]:
    corrected_p_values, significant = _fdr_correct(
        p_values
    )

    return [
        ChannelStatisticRow(
            comparison=row.comparison,
            comparison_slug=row.comparison_slug,
            channel=row.channel,
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
            significant_fdr=bool(is_significant),
            direction=row.direction,
        )
        for row, p_value_fdr, is_significant in zip(
            rows,
            corrected_p_values,
            significant,
            strict=True,
        )
    ]


def _fdr_correct(
    p_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Benjamini-Hochberg FDR to finite p-values.
    """
    p_values_array = np.asarray(
        p_values,
        dtype=np.float64,
    )

    corrected = np.full_like(
        p_values_array,
        np.nan,
    )
    significant = np.zeros(
        p_values_array.shape,
        dtype=bool,
    )

    finite_mask = np.isfinite(
        p_values_array
    )

    if np.any(
        finite_mask
    ):
        reject, corrected_values, _, _ = multipletests(
            p_values_array[
                finite_mask
            ],
            alpha=ALPHA,
            method="fdr_bh",
        )

        corrected[
            finite_mask
        ] = corrected_values
        significant[
            finite_mask
        ] = reject

    return corrected, significant


def _direction_label(
    comparison: ChannelComparison,
    mean_difference: float,
) -> str:
    """
    Build a readable direction from the A-B paired mean difference sign.
    """
    if not np.isfinite(
        mean_difference
    ):
        return "not available"

    operator = (
        ">"
        if mean_difference > 0
        else "<"
        if mean_difference < 0
        else "="
    )

    return (
        f"{comparison.left_label} "
        f"{operator} "
        f"{comparison.right_label}"
    )


def _mean_sd(
    mean: float,
    std: float,
) -> str:
    return (
        f"{mean:.4f}+/-{std:.4f}"
    )
