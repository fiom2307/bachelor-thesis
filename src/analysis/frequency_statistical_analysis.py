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
    FREQUENCY_BANDS as EEGNET_SOURCE_FREQUENCY_BANDS,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS, CLASS_NAMES
from src.utils.paths import (
    get_csp_fold_model_path,
    get_frequency_domain_shap_values_path,
    get_frequency_statistical_classwise_results_path,
    get_frequency_statistical_overall_results_path,
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

STATISTICAL_FREQUENCY_BANDS: tuple[
    tuple[str, float, float],
    ...,
] = (
    ("Mu", 8.0, 13.0),
    ("Low beta", 13.0, 20.0),
    ("High beta", 20.0, 30.0),
)


@dataclass(frozen=True)
class SubjectFrequencyProfile:
    subject: int
    model: ModelName
    condition: TrialSelection
    class_values: np.ndarray

    @property
    def values(self) -> np.ndarray:
        return _class_balanced_relative_profile(
            self.class_values
        )


@dataclass(frozen=True)
class FrequencyComparison:
    name: str
    slug: str
    left_label: str
    right_label: str
    left_key: tuple[ModelName, TrialSelection]
    right_key: tuple[ModelName, TrialSelection]


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
    class_name: str | None = None


COMPARISONS: tuple[
    FrequencyComparison,
    ...,
] = (
    FrequencyComparison(
        name="CSP+LDA correct vs EEGNet correct",
        slug="csp_lda_correct_vs_eegnet_correct",
        left_label="CSP+LDA correct",
        right_label="EEGNet correct",
        left_key=("csp", "correct"),
        right_key=("eegnet", "correct"),
    ),
    FrequencyComparison(
        name="CSP+LDA correct vs CSP+LDA incorrect",
        slug="csp_lda_correct_vs_csp_lda_incorrect",
        left_label="CSP+LDA correct",
        right_label="CSP+LDA incorrect",
        left_key=("csp", "correct"),
        right_key=("csp", "incorrect"),
    ),
    FrequencyComparison(
        name="EEGNet correct vs EEGNet incorrect",
        slug="eegnet_correct_vs_eegnet_incorrect",
        left_label="EEGNet correct",
        right_label="EEGNet incorrect",
        left_key=("eegnet", "correct"),
        right_key=("eegnet", "incorrect"),
    ),
)


def run_frequency_statistical_analysis() -> tuple[
    list[SubjectFrequencyProfile],
    dict[str, list[FrequencyStatisticRow]],
]:
    """
    Run subject-level frequency relevance statistics.
    """
    profiles = collect_subject_frequency_profiles()
    rows_by_output = compute_frequency_statistics(
        profiles
    )

    save_frequency_statistics(
        rows_by_output
    )

    print_frequency_statistics(
        rows_by_output
    )

    return profiles, rows_by_output


def collect_subject_frequency_profiles() -> list[
    SubjectFrequencyProfile
]:
    """
    Build class-wise relative profiles per subject/model/condition.
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
) -> dict[str, list[FrequencyStatisticRow]]:
    """
    Compute overall and class-wise Wilcoxon tests for each comparison.
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
        overall_rows = _compute_overall_statistics(
            profile_lookup=profile_lookup,
            comparison=comparison,
        )

        classwise_rows = _compute_classwise_statistics(
            profile_lookup=profile_lookup,
            comparison=comparison,
        )

        rows_by_output[
            _overall_output_key(
                comparison
            )
        ] = overall_rows

        rows_by_output[
            _classwise_output_key(
                comparison
            )
        ] = classwise_rows

    return rows_by_output


def save_frequency_statistics(
    rows_by_output: dict[str, list[FrequencyStatisticRow]],
) -> list[Path]:
    """
    Save overall_statistics.csv and classwise_statistics.csv per comparison.
    """
    output_paths = []

    for comparison in COMPARISONS:
        overall_path = get_frequency_statistical_overall_results_path(
            comparison.slug
        )

        _save_frequency_statistics_rows(
            rows=rows_by_output[
                _overall_output_key(
                    comparison
                )
            ],
            output_file=overall_path,
            include_class=False,
        )

        output_paths.append(
            overall_path
        )

        classwise_path = (
            get_frequency_statistical_classwise_results_path(
                comparison.slug
            )
        )

        _save_frequency_statistics_rows(
            rows=rows_by_output[
                _classwise_output_key(
                    comparison
                )
            ],
            output_file=classwise_path,
            include_class=True,
        )

        output_paths.append(
            classwise_path
        )

    return output_paths


def print_frequency_statistics(
    rows_by_output: dict[str, list[FrequencyStatisticRow]],
) -> None:
    """
    Print a compact console summary grouped by comparison.
    """
    for comparison in COMPARISONS:
        print()
        print("=" * 70)
        print(comparison.name)
        print("=" * 70)
        print(
            f"{'Scope':<12} "
            f"{'Band':<10} "
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
            _print_frequency_row(
                scope="Overall",
                row=row,
            )

        for row in rows_by_output[
            _classwise_output_key(
                comparison
            )
        ]:
            _print_frequency_row(
                scope=row.class_name or "",
                row=row,
            )


def _compute_overall_statistics(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectFrequencyProfile,
    ],
    comparison: FrequencyComparison,
) -> list[FrequencyStatisticRow]:
    rows = []
    raw_p_values = []

    for band_index, (band_name, band_low, band_high) in enumerate(
        STATISTICAL_FREQUENCY_BANDS
    ):
        left_values, right_values = _paired_overall_band_values(
            profile_lookup=profile_lookup,
            comparison=comparison,
            band_index=band_index,
        )

        row = _compute_statistic_row(
            comparison=comparison,
            frequency_band=band_name,
            band_low=band_low,
            band_high=band_high,
            left_values=left_values,
            right_values=right_values,
        )

        rows.append(
            row
        )

        raw_p_values.append(
            row.p_value
        )

    return _with_fdr_correction(
        rows,
        raw_p_values,
    )


def _compute_classwise_statistics(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectFrequencyProfile,
    ],
    comparison: FrequencyComparison,
) -> list[FrequencyStatisticRow]:
    corrected_rows = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        class_rows = []
        raw_p_values = []

        for band_index, (band_name, band_low, band_high) in enumerate(
            STATISTICAL_FREQUENCY_BANDS
        ):
            left_values, right_values = _paired_class_band_values(
                profile_lookup=profile_lookup,
                comparison=comparison,
                class_index=class_index,
                band_index=band_index,
            )

            row = _compute_statistic_row(
                comparison=comparison,
                frequency_band=band_name,
                band_low=band_low,
                band_high=band_high,
                left_values=left_values,
                right_values=right_values,
                class_name=class_name,
            )

            class_rows.append(
                row
            )

            raw_p_values.append(
                row.p_value
            )

        corrected_rows.extend(
            _with_fdr_correction(
                class_rows,
                raw_p_values,
            )
        )

    return corrected_rows


def _compute_statistic_row(
    comparison: FrequencyComparison,
    frequency_band: str,
    band_low: float,
    band_high: float,
    left_values: np.ndarray,
    right_values: np.ndarray,
    class_name: str | None = None,
) -> FrequencyStatisticRow:
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

    return FrequencyStatisticRow(
        comparison=comparison.name,
        comparison_slug=comparison.slug,
        class_name=class_name,
        frequency_band=frequency_band,
        band_low=band_low,
        band_high=band_high,
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
            frequency_bands=_statistical_band_ranges(),
        )

        profiles.append(
            SubjectFrequencyProfile(
                subject=subject,
                model="csp",
                condition=condition,
                class_values=_relative_class_profiles(
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

    if result.frequency_bands != EEGNET_SOURCE_FREQUENCY_BANDS:
        raise ValueError(
            "Saved SHAP frequency bands do not match "
            "the configured EEGNet source frequency bands."
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

        source_class_relevance = _mapping_to_class_matrix(
            frequency_relevance,
            n_bands=len(EEGNET_SOURCE_FREQUENCY_BANDS),
        )

        band_relevance = _aggregate_source_bands_to_bands(
            values=source_class_relevance,
            source_bands=EEGNET_SOURCE_FREQUENCY_BANDS,
            target_bands=_statistical_band_ranges(),
        )

        profiles.append(
            SubjectFrequencyProfile(
                subject=subject,
                model="eegnet",
                condition=condition,
                class_values=_relative_class_profiles(
                    band_relevance
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
    frequency_bands: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """
    Aggregate dense CSP FFT bins into statistical bands.
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
            len(frequency_bands),
        ),
        np.nan,
        dtype=np.float64,
    )

    for band_index, (low, high) in enumerate(
        frequency_bands
    ):
        if band_index == len(frequency_bands) - 1:
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


def _aggregate_source_bands_to_bands(
    values: np.ndarray,
    source_bands: tuple[tuple[float, float], ...],
    target_bands: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """
    Aggregate saved EEGNet 2-Hz SHAP bands into statistical bands.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    band_values = np.full(
        (
            values.shape[0],
            len(target_bands),
        ),
        np.nan,
        dtype=np.float64,
    )

    for target_index, (target_low, target_high) in enumerate(
        target_bands
    ):
        weighted_values = np.zeros(
            values.shape[0],
            dtype=np.float64,
        )

        has_overlap = False

        for source_index, (source_low, source_high) in enumerate(
            source_bands
        ):
            overlap = max(
                0.0,
                min(target_high, source_high)
                - max(target_low, source_low),
            )

            if overlap <= 0:
                continue

            has_overlap = True
            source_width = source_high - source_low

            weighted_values += (
                values[
                    :,
                    source_index,
                ]
                * (overlap / source_width)
            )

        if has_overlap:
            band_values[
                :,
                target_index,
            ] = weighted_values

    return band_values


def _mapping_to_class_matrix(
    relevance: dict[int, np.ndarray],
    n_bands: int,
) -> np.ndarray:
    """
    Convert class-id keyed relevance to a class x band matrix.
    """
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
                "value per source frequency band."
            )

        matrix[
            class_index
        ] = values

    return matrix


def _relative_class_profiles(
    class_relevance: np.ndarray,
) -> np.ndarray:
    """
    Normalize each class profile to sum to one.
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

    return normalized


def _class_balanced_relative_profile(
    class_values: np.ndarray,
) -> np.ndarray:
    """
    Average the four class-normalized profiles with equal class weight.
    """
    if (
        class_values.shape[0]
        != len(CLASS_LABELS)
    ):
        raise ValueError(
            "Class-balanced profiles require all motor imagery classes."
        )

    if np.any(
        ~np.isfinite(
            class_values
        )
    ):
        return np.full(
            class_values.shape[1],
            np.nan,
            dtype=np.float64,
        )

    return np.mean(
        class_values,
        axis=0,
    )


def _paired_overall_band_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectFrequencyProfile,
    ],
    comparison: FrequencyComparison,
    band_index: int,
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

        left_value = left_profile.values[
            band_index
        ]

        right_value = right_profile.values[
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


def _paired_class_band_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        SubjectFrequencyProfile,
    ],
    comparison: FrequencyComparison,
    class_index: int,
    band_index: int,
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
            class_index,
            band_index,
        ]

        right_value = right_profile.class_values[
            class_index,
            band_index,
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
    rows: list[FrequencyStatisticRow],
    p_values: list[float],
) -> list[FrequencyStatisticRow]:
    corrected_p_values, significant = _fdr_correct(
        p_values
    )

    return [
        FrequencyStatisticRow(
            comparison=row.comparison,
            comparison_slug=row.comparison_slug,
            class_name=row.class_name,
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


def _save_frequency_statistics_rows(
    rows: list[FrequencyStatisticRow],
    output_file: str | Path,
    include_class: bool,
) -> Path:
    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "comparison",
    ]

    if include_class:
        fieldnames.append(
            "class"
        )

    fieldnames.extend([
        "frequency_band",
        "band_low",
        "band_high",
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
    ])

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            output_row = {
                "comparison": row.comparison,
                "frequency_band": row.frequency_band,
                "band_low": _format_float(row.band_low),
                "band_high": _format_float(row.band_high),
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
            }

            if include_class:
                output_row[
                    "class"
                ] = row.class_name

            writer.writerow(
                output_row
            )

    return output_file


def _direction_label(
    comparison: FrequencyComparison,
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


def _statistical_band_ranges() -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            low,
            high,
        )
        for _, low, high in STATISTICAL_FREQUENCY_BANDS
    )


def _overall_output_key(
    comparison: FrequencyComparison,
) -> str:
    return f"{comparison.slug}:overall"


def _classwise_output_key(
    comparison: FrequencyComparison,
) -> str:
    return f"{comparison.slug}:classwise"


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


def _print_frequency_row(
    scope: str,
    row: FrequencyStatisticRow,
) -> None:
    marker = (
        "*"
        if row.significant_fdr
        else ""
    )

    print(
        f"{scope:<12} "
        f"{row.frequency_band:<10} "
        f"{_mean_sd(row.mean_a, row.std_a):<28} "
        f"{_mean_sd(row.mean_b, row.std_b):<28} "
        f"{row.mean_difference:>8.4f} "
        f"{row.p_value_fdr:>10.4g}"
        f"{marker}"
    )


def _format_float(
    value: float,
) -> str:
    if not np.isfinite(
        value
    ):
        return ""

    return f"{value:.10g}"
