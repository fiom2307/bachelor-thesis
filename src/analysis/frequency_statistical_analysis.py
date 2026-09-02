from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.analysis.csp_pattern_analysis.frequency_relevance import (
    aggregate_trial_frequency_relevance,
    compute_trial_frequency_relevance,
)
from src.analysis.shap_analysis import (
    compute_frequency_shap_relevance,
    load_frequency_domain_shap_result,
)
from src.analysis.shap_analysis.frequency_domain.shap_analysis import (
    FREQUENCY_BANDS,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS
from src.utils.paths import (
    get_csp_fold_model_path,
    get_frequency_domain_shap_values_path,
    get_frequency_statistical_profiles_path,
    get_frequency_statistical_results_path,
    get_lda_fold_model_path,
    get_subject_name,
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
SFREQ = 250.0
FMIN = 8.0
FMAX = 30.0
ALPHA = 0.05


@dataclass(frozen=True)
class SubjectFrequencyProfile:
    subject: int
    model: ModelName
    condition: TrialSelection
    values: np.ndarray


@dataclass(frozen=True)
class FrequencyComparison:
    name: str
    slug: str
    left_label: str
    right_label: str
    left_key: tuple[ModelName, TrialSelection]
    right_key: tuple[ModelName, TrialSelection]
    difference_label: str


@dataclass(frozen=True)
class FrequencyStatisticRow:
    comparison: str
    comparison_slug: str
    frequency_band: str
    band_low: float
    band_high: float
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
    FrequencyComparison,
    ...,
] = (
    FrequencyComparison(
        name="CSP+LDA CORRECT vs EEGNet CORRECT",
        slug="csp_lda_correct_vs_eegnet_correct",
        left_label="CSP correct",
        right_label="EEGNet correct",
        left_key=("csp", "correct"),
        right_key=("eegnet", "correct"),
        difference_label="EEGNet correct - CSP correct",
    ),
    FrequencyComparison(
        name="EEGNet CORRECT vs EEGNet INCORRECT",
        slug="eegnet_correct_vs_eegnet_incorrect",
        left_label="EEGNet incorrect",
        right_label="EEGNet correct",
        left_key=("eegnet", "incorrect"),
        right_key=("eegnet", "correct"),
        difference_label="EEGNet correct - EEGNet incorrect",
    ),
    FrequencyComparison(
        name="CSP+LDA CORRECT vs CSP+LDA INCORRECT",
        slug="csp_lda_correct_vs_csp_lda_incorrect",
        left_label="CSP incorrect",
        right_label="CSP correct",
        left_key=("csp", "incorrect"),
        right_key=("csp", "correct"),
        difference_label="CSP correct - CSP incorrect",
    ),
)


def run_frequency_statistical_analysis() -> tuple[
    list[SubjectFrequencyProfile],
    list[FrequencyStatisticRow],
]:
    """
    Run the subject-level frequency relevance statistical analysis.
    """
    profiles = collect_subject_frequency_profiles()
    rows = compute_frequency_statistics(profiles)

    save_frequency_profiles(
        profiles
    )

    save_frequency_statistics(
        rows
    )

    print_frequency_statistics(
        rows
    )

    return profiles, rows


def collect_subject_frequency_profiles() -> list[
    SubjectFrequencyProfile
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


def compute_frequency_statistics(
    profiles: list[SubjectFrequencyProfile],
) -> list[FrequencyStatisticRow]:
    """
    Compute paired Wilcoxon tests and descriptive statistics per band.
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

        for band_index, (band_low, band_high) in enumerate(
            FREQUENCY_BANDS
        ):
            left_values, right_values = _paired_band_values(
                profile_lookup=profile_lookup,
                comparison=comparison,
                band_index=band_index,
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

            comparison_rows.append(
                FrequencyStatisticRow(
                    comparison=comparison.name,
                    comparison_slug=comparison.slug,
                    frequency_band=_format_band(
                        band_low,
                        band_high,
                    ),
                    band_low=band_low,
                    band_high=band_high,
                    n=len(differences),
                    mean_a=_nanmean(left_values),
                    std_a=_nanstd(left_values),
                    median_a=_nanmedian(left_values),
                    mean_b=_nanmean(right_values),
                    std_b=_nanstd(right_values),
                    median_b=_nanmedian(right_values),
                    mean_difference=_nanmean(differences),
                    median_difference=_nanmedian(differences),
                    wilcoxon_statistic=statistic,
                    p_value=p_value,
                    p_value_fdr=np.nan,
                    significant_fdr=False,
                    direction=_direction_label(
                        comparison=comparison,
                        mean_difference=_nanmean(
                            differences
                        ),
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
                FrequencyStatisticRow(
                    comparison=row.comparison,
                    comparison_slug=row.comparison_slug,
                    frequency_band=row.frequency_band,
                    band_low=row.band_low,
                    band_high=row.band_high,
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


def save_frequency_profiles(
    profiles: list[SubjectFrequencyProfile],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save subject-level class-balanced relative profiles.
    """
    if output_file is None:
        output_file = get_frequency_statistical_profiles_path()

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
            "frequency_band",
            "relative_relevance",
        ])

        for profile in profiles:
            subject_name = get_subject_name(
                profile.subject
            )

            for band, value in zip(
                FREQUENCY_BANDS,
                profile.values,
                strict=True,
            ):
                writer.writerow([
                    subject_name,
                    profile.model,
                    profile.condition,
                    _format_band(*band),
                    _format_float(value),
                ])

    return output_file


def save_frequency_statistics(
    rows: list[FrequencyStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save statistical test results.
    """
    if output_file is None:
        output_file = get_frequency_statistical_results_path()

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
                "frequency_band",
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
                "frequency_band": row.frequency_band,
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


def print_frequency_statistics(
    rows: list[FrequencyStatisticRow],
) -> None:
    """
    Print a compact console summary grouped by comparison.
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
        print(comparison.name)
        print("=" * 70)
        print(
            f"{'Band':<8} "
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
                f"{row.frequency_band:<8} "
                f"{_mean_sd(row.mean_a, row.std_a):<24} "
                f"{_mean_sd(row.mean_b, row.std_b):<26} "
                f"{row.mean_difference:>8.4f} "
                f"{row.p_value:>10.4g} "
                f"{row.p_value_fdr:>10.4g}"
                f"{marker}"
            )


def _load_csp_subject_profiles(
    subject: int,
) -> list[SubjectFrequencyProfile]:
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

    result = compute_trial_frequency_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
        sfreq=SFREQ,
        fmin=FMIN,
        fmax=FMAX,
    )

    profiles = []

    for condition, mask in (
        ("correct", result.correct_mask),
        ("incorrect", result.incorrect_mask),
    ):
        class_relevance, _ = aggregate_trial_frequency_relevance(
            result=result,
            mask=mask,
        )

        band_relevance = _aggregate_frequency_bins_to_bands(
            values=class_relevance,
            frequencies=result.frequencies,
        )

        profiles.append(
            SubjectFrequencyProfile(
                subject=subject,
                model="csp",
                condition=condition,
                values=_class_balanced_relative_profile(
                    band_relevance
                ),
            )
        )

    return profiles


def _load_eegnet_subject_profiles(
    subject: int,
) -> list[SubjectFrequencyProfile]:
    """
    Load saved frequency-domain SHAP values and summarize relevance.
    """
    shap_file = get_frequency_domain_shap_values_path(
        subject
    )

    if not shap_file.exists():
        raise FileNotFoundError(
            "Saved frequency-domain SHAP values are missing: "
            f"{shap_file}"
        )

    result = load_frequency_domain_shap_result(
        shap_file
    )

    if result.frequency_bands != FREQUENCY_BANDS:
        raise ValueError(
            "Saved SHAP frequency bands do not match "
            "the configured frequency bands."
        )

    profiles = []

    for condition, mask in (
        ("correct", result.correct_mask),
        ("incorrect", result.incorrect_mask),
    ):
        frequency_relevance = compute_frequency_shap_relevance(
            shap_values=result.values,
            labels=result.labels,
            trial_mask=mask,
        )

        class_relevance = _mapping_to_class_matrix(
            frequency_relevance
        )

        profiles.append(
            SubjectFrequencyProfile(
                subject=subject,
                model="eegnet",
                condition=condition,
                values=_class_balanced_relative_profile(
                    class_relevance
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


def _aggregate_frequency_bins_to_bands(
    values: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """
    Aggregate dense CSP FFT bins into the configured two-Hz bands.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    frequencies = np.asarray(
        frequencies,
        dtype=np.float64,
    )

    band_values = np.full(
        (
            values.shape[0],
            len(FREQUENCY_BANDS),
        ),
        np.nan,
        dtype=np.float64,
    )

    for band_index, (low, high) in enumerate(
        FREQUENCY_BANDS
    ):
        if band_index == len(FREQUENCY_BANDS) - 1:
            frequency_mask = (
                (frequencies >= low)
                & (frequencies <= high)
            )
        else:
            frequency_mask = (
                (frequencies >= low)
                & (frequencies < high)
            )

        if not np.any(
            frequency_mask
        ):
            continue

        band_values[
            :,
            band_index,
        ] = np.nansum(
            values[
                :,
                frequency_mask,
            ],
            axis=1,
        )

    return band_values


def _mapping_to_class_matrix(
    relevance: dict[int, np.ndarray],
) -> np.ndarray:
    """
    Convert class-id keyed relevance to a class x band matrix.
    """
    n_bands = len(
        FREQUENCY_BANDS
    )

    matrix = np.full(
        (
            len(CLASS_LABELS),
            n_bands,
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
            n_bands,
        ):
            raise ValueError(
                "Frequency relevance must contain one "
                "value per configured frequency band."
            )

        matrix[
            class_index
        ] = values

    return matrix


def _class_balanced_relative_profile(
    class_relevance: np.ndarray,
) -> np.ndarray:
    """
    Normalize each class profile to sum to one, then average classes.
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


def _paired_band_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        np.ndarray,
    ],
    comparison: FrequencyComparison,
    band_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return paired left/right values for one comparison and band.
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
            band_index
        ]

        right_value = right_profile[
            band_index
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
    comparison: FrequencyComparison,
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


def _format_band(
    low: float,
    high: float,
) -> str:
    return f"{low:g}-{high:g}"


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
