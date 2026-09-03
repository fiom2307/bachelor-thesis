from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.analysis.csp_pattern_analysis.channel_relevance import (
    aggregate_trial_channel_relevance,
    compute_trial_channel_relevance,
)
from src.analysis.shap_analysis import (
    compute_channel_shap_relevance,
    load_time_domain_shap_result,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS
from src.data.preprocessing import BCI_2A_CHANNEL_NAMES
from src.utils.paths import (
    get_channel_statistical_profiles_path,
    get_channel_statistical_results_path,
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_subject_name,
    get_time_domain_shap_values_path,
)


TrialSelection = Literal[
    "correct",
    "incorrect",
]

ModelName = Literal[
    "csp",
    "eegnet",
]


SUBJECTS = range(1, 10)
N_FOLDS = 5
ALPHA = 0.05

CHANNEL_NAMES = BCI_2A_CHANNEL_NAMES


@dataclass(frozen=True)
class SubjectChannelProfile:
    subject: int
    model: ModelName
    condition: TrialSelection
    values: np.ndarray


@dataclass(frozen=True)
class ChannelComparison:
    name: str
    slug: str
    left_label: str
    right_label: str
    left_key: tuple[ModelName, TrialSelection]
    right_key: tuple[ModelName, TrialSelection]
    difference_label: str


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


COMPARISONS: tuple[
    ChannelComparison,
    ...,
] = (
    ChannelComparison(
        name="CSP+LDA CORRECT vs EEGNet CORRECT",
        slug="csp_lda_correct_vs_eegnet_correct",
        left_label="CSP correct",
        right_label="EEGNet correct",
        left_key=("csp", "correct"),
        right_key=("eegnet", "correct"),
        difference_label="EEGNet correct - CSP correct",
    ),
    ChannelComparison(
        name="EEGNet CORRECT vs EEGNet INCORRECT",
        slug="eegnet_correct_vs_eegnet_incorrect",
        left_label="EEGNet incorrect",
        right_label="EEGNet correct",
        left_key=("eegnet", "incorrect"),
        right_key=("eegnet", "correct"),
        difference_label="EEGNet correct - EEGNet incorrect",
    ),
    ChannelComparison(
        name="CSP+LDA CORRECT vs CSP+LDA INCORRECT",
        slug="csp_lda_correct_vs_csp_lda_incorrect",
        left_label="CSP incorrect",
        right_label="CSP correct",
        left_key=("csp", "incorrect"),
        right_key=("csp", "correct"),
        difference_label="CSP correct - CSP incorrect",
    ),
)


def run_channel_statistical_analysis() -> tuple[
    list[SubjectChannelProfile],
    list[ChannelStatisticRow],
]:
    """
    Run the subject-level channel relevance statistical analysis.
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


def collect_subject_channel_profiles() -> list[
    SubjectChannelProfile
]:
    """
    Build one class-balanced relative profile per subject/model/condition.
    """
    profiles = []

    for subject in SUBJECTS:
        profiles.extend(
            _load_csp_subject_profiles(
                subject
            )
        )

        profiles.extend(
            _load_eegnet_subject_profiles(
                subject
            )
        )

    return profiles


def compute_channel_statistics(
    profiles: list[SubjectChannelProfile],
) -> list[ChannelStatisticRow]:
    """
    Compute paired Wilcoxon tests and descriptives per channel.
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

            differences = (
                right_values
                - left_values
            )

            statistic, p_value = _paired_wilcoxon(
                differences
            )

            raw_p_values.append(
                p_value
            )

            mean_difference = _nanmean(
                differences
            )

            comparison_rows.append(
                ChannelStatisticRow(
                    comparison=comparison.name,
                    comparison_slug=comparison.slug,
                    channel=channel_name,
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
            )

        corrected_p_values, significant = _fdr_correct(
            raw_p_values
        )

        for row, p_value_fdr, is_significant in zip(
            comparison_rows,
            corrected_p_values,
            significant,
            strict=True,
        ):
            rows.append(
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
                    wilcoxon_statistic=(
                        row.wilcoxon_statistic
                    ),
                    p_value=row.p_value,
                    p_value_fdr=p_value_fdr,
                    significant_fdr=bool(
                        is_significant
                    ),
                    direction=row.direction,
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
    Save channel statistical test results.
    """
    if output_file is None:
        output_file = get_channel_statistical_results_path()

    output_file = Path(
        output_file
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
    rows_by_comparison = {
        comparison.name: [
            row
            for row in rows
            if row.comparison == comparison.name
        ]
        for comparison in COMPARISONS
    }

    for comparison in COMPARISONS:
        print()
        print("=" * 70)
        print(
            f"{comparison.name} - CHANNEL RELEVANCE"
        )
        print("=" * 70)
        print(
            f"{'Channel':<8} "
            f"{comparison.left_label + ' mean+/-SD':<24} "
            f"{comparison.right_label + ' mean+/-SD':<26} "
            f"{'Delta':>8} "
            f"{'p':>10} "
            f"{'p_FDR':>10}"
        )

        for row in rows_by_comparison[
            comparison.name
        ]:
            marker = (
                "*"
                if row.significant_fdr
                else ""
            )

            print(
                f"{row.channel:<8} "
                f"{_mean_sd(row.mean_a, row.std_a):<24} "
                f"{_mean_sd(row.mean_b, row.std_b):<26} "
                f"{row.mean_difference:>8.4f} "
                f"{row.p_value:>10.4g} "
                f"{row.p_value_fdr:>10.4g}"
                f"{marker}"
            )


def _load_csp_subject_profiles(
    subject: int,
) -> list[SubjectChannelProfile]:
    """
    Compute subject-level CSP+LDA profiles using existing relevance code.
    """
    csps, ldas = _load_subject_models(
        subject
    )

    subject_data = get_data_for_subject(
        subject
    )

    if subject_data is None:
        raise FileNotFoundError(
            "Could not load data for "
            f"{get_subject_name(subject)}."
        )

    _, _, x_eval, y_eval = subject_data

    result = compute_trial_channel_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
    )

    profiles = []

    for condition, mask in (
        ("correct", result.correct_mask),
        ("incorrect", result.incorrect_mask),
    ):
        class_relevance, _ = aggregate_trial_channel_relevance(
            result=result,
            mask=mask,
        )

        profiles.append(
            SubjectChannelProfile(
                subject=subject,
                model="csp",
                condition=condition,
                values=_class_balanced_relative_profile(
                    class_relevance
                ),
            )
        )

    return profiles


def _load_eegnet_subject_profiles(
    subject: int,
) -> list[SubjectChannelProfile]:
    """
    Load saved time-domain SHAP values and summarize channel relevance.
    """
    shap_file = get_time_domain_shap_values_path(
        subject
    )

    if not shap_file.exists():
        raise FileNotFoundError(
            "Saved time-domain SHAP values are missing: "
            f"{shap_file}"
        )

    result = load_time_domain_shap_result(
        shap_file
    )

    profiles = []

    for condition, mask in (
        ("correct", result.correct_mask),
        ("incorrect", result.incorrect_mask),
    ):
        class_relevance = _compute_class_shap_relevance(
            shap_values=result.values,
            labels=result.labels,
            trial_mask=mask,
        )

        channel_relevance = compute_channel_shap_relevance(
            class_relevance
        )

        profiles.append(
            SubjectChannelProfile(
                subject=subject,
                model="eegnet",
                condition=condition,
                values=_class_balanced_relative_profile(
                    _mapping_to_class_matrix(
                        channel_relevance
                    )
                ),
            )
        )

    return profiles


def _load_subject_models(
    subject: int,
):
    """
    Load all CSP and LDA fold models for one subject.
    """
    csps = []
    ldas = []

    for fold in _get_fold_numbers(
        subject
    ):
        csps.append(
            joblib.load(
                get_csp_fold_model_path(
                    subject,
                    fold,
                )
            )
        )

        ldas.append(
            joblib.load(
                get_lda_fold_model_path(
                    subject,
                    fold,
                )
            )
        )

    return csps, ldas


def _get_fold_numbers(
    subject: int,
) -> list[int]:
    """
    Determine whether saved folds are numbered 0-4 or 1-5.
    """
    for folds in (
        list(range(1, N_FOLDS + 1)),
        list(range(N_FOLDS)),
    ):
        if all(
            get_csp_fold_model_path(
                subject,
                fold,
            ).exists()
            and get_lda_fold_model_path(
                subject,
                fold,
            ).exists()
            for fold in folds
        ):
            return folds

    raise FileNotFoundError(
        "Could not find all CSP+LDA fold models for "
        f"{get_subject_name(subject)}."
    )


def _compute_class_shap_relevance(
    shap_values: np.ndarray,
    labels: np.ndarray,
    trial_mask: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute class-wise mean absolute SHAP relevance for selected trials.
    """
    class_relevance = {}

    for class_id in CLASS_LABELS:
        class_mask = (
            (labels == class_id)
            & trial_mask
        )

        if not np.any(
            class_mask
        ):
            continue

        class_relevance[
            class_id
        ] = np.abs(
            shap_values[
                class_mask
            ]
        ).mean(
            axis=0
        )

    return class_relevance


def _mapping_to_class_matrix(
    relevance: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Convert class-id keyed relevance to a class x channel matrix.
    """
    matrix = np.full(
        (
            len(CLASS_LABELS),
            len(CHANNEL_NAMES),
        ),
        np.nan,
        dtype=np.float64,
    )

    for class_index, class_id in enumerate(
        CLASS_LABELS
    ):
        if class_id not in relevance:
            continue

        values = np.asarray(
            relevance[class_id],
            dtype=np.float64,
        )

        if values.shape != (
            len(CHANNEL_NAMES),
        ):
            raise ValueError(
                "Channel relevance must contain one value "
                "per configured EEG channel."
            )

        matrix[
            class_index
        ] = values

    return matrix


def _class_balanced_relative_profile(
    class_relevance: np.ndarray,
) -> np.ndarray:
    """
    Normalize each class over channels, then average classes equally.
    """
    class_relevance = np.asarray(
        class_relevance,
        dtype=np.float64,
    )

    normalized = np.full_like(
        class_relevance,
        np.nan,
        dtype=np.float64,
    )

    for class_index in range(
        class_relevance.shape[0]
    ):
        values = class_relevance[
            class_index
        ]

        denominator = np.nansum(
            values
        )

        if (
            not np.isfinite(
                denominator
            )
            or denominator <= 0
        ):
            continue

        normalized[
            class_index
        ] = values / denominator

    if np.all(
        np.isnan(
            normalized
        )
    ):
        return np.full(
            class_relevance.shape[1],
            np.nan,
            dtype=np.float64,
        )

    return np.nanmean(
        normalized,
        axis=0,
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
    Compute a paired Wilcoxon signed-rank test from paired differences.
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
        differences
    )

    return (
        float(statistic),
        float(p_value),
    )


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
    Build a readable direction from the paired mean difference sign.
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
        f"{comparison.right_label} "
        f"{operator} "
        f"{comparison.left_label}"
    )


def _nanmean(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return np.nan

    return float(
        np.nanmean(
            values
        )
    )


def _nanmedian(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return np.nan

    return float(
        np.nanmedian(
            values
        )
    )


def _nanstd(
    values: np.ndarray,
) -> float:
    if len(values) < 2:
        return np.nan

    return float(
        np.nanstd(
            values,
            ddof=1,
        )
    )


def _mean_sd(
    mean: float,
    std: float,
) -> str:
    return (
        f"{mean:.4f}+/-{std:.4f}"
    )


def _format_float(
    value: float,
) -> str:
    if not np.isfinite(
        value
    ):
        return ""

    return f"{value:.10g}"
