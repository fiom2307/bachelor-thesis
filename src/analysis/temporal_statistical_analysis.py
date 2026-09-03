from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

from src.analysis.csp_pattern_analysis.temporal_relevance import (
    aggregate_trial_temporal_relevance,
    compute_trial_temporal_relevance,
)
from src.analysis.shap_analysis import (
    compute_temporal_shap_relevance,
    load_time_domain_shap_result,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS, CLASS_NAMES
from src.utils.config import EPOCH_TMIN
from src.utils.paths import (
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_subject_name,
    get_temporal_classwise_csp_vs_eegnet_correct_results_path,
    get_temporal_classwise_statistical_results_path,
    get_temporal_csp_lda_right_hand_early_late_gee_results_path,
    get_temporal_csp_lda_right_hand_early_late_gee_trials_path,
    get_temporal_left_hand_early_gee_results_path,
    get_temporal_left_hand_early_gee_trials_path,
    get_temporal_statistical_profiles_path,
    get_temporal_statistical_results_path,
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
SFREQ = 250.0
ALPHA = 0.05
NORMALIZATION_WINDOW = (0.5, 4.0)
LEFT_HAND_LABEL = 0
RIGHT_HAND_LABEL = 1

TEMPORAL_WINDOWS: tuple[
    tuple[str, float, float],
    ...,
] = (
    ("Early", 0.5, 1.5),
    ("Middle", 1.5, 2.5),
    ("Late", 2.5, 4.0),
)


@dataclass(frozen=True)
class SubjectTemporalProfile:
    subject: int
    model: ModelName
    condition: TrialSelection
    values: np.ndarray


@dataclass(frozen=True)
class TemporalComparison:
    name: str
    slug: str
    left_label: str
    right_label: str
    left_key: tuple[ModelName, TrialSelection]
    right_key: tuple[ModelName, TrialSelection]
    difference_label: str


@dataclass(frozen=True)
class TemporalStatisticRow:
    comparison: str
    comparison_slug: str
    temporal_window: str
    start_time: float
    end_time: float
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


@dataclass(frozen=True)
class SubjectClassTemporalProfile:
    subject: int
    class_label: int
    class_name: str
    condition: TrialSelection
    values: np.ndarray


@dataclass(frozen=True)
class SubjectModelClassTemporalProfile:
    subject: int
    model: ModelName
    class_label: int
    class_name: str
    condition: TrialSelection
    values: np.ndarray


@dataclass(frozen=True)
class ClassTemporalStatisticRow:
    class_label: int
    class_name: str
    temporal_window: str
    start_time: float
    end_time: float
    n: int
    correct_mean: float
    correct_std: float
    incorrect_mean: float
    incorrect_std: float
    mean_difference: float
    wilcoxon_statistic: float
    p_value: float
    p_value_fdr: float
    significant_fdr: bool
    direction: str


@dataclass(frozen=True)
class ClassModelTemporalStatisticRow:
    class_label: int
    class_name: str
    temporal_window: str
    start_time: float
    end_time: float
    n: int
    csp_correct_mean: float
    csp_correct_std: float
    eegnet_correct_mean: float
    eegnet_correct_std: float
    mean_difference: float
    wilcoxon_statistic: float
    p_value: float
    p_value_fdr: float
    significant_fdr: bool
    direction: str


@dataclass(frozen=True)
class LeftHandEarlyRelevanceTrial:
    subject: int
    trial_index: int
    early_relevance: float
    correct: int


@dataclass(frozen=True)
class LeftHandEarlyRelevanceGeeRow:
    coefficient: float
    standard_error: float
    odds_ratio: float
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float
    p_value: float
    n_subjects: int
    n_left_hand_trials: int
    interpretation: str


@dataclass(frozen=True)
class CspLdaRightHandEarlyLateContrastGeeTrial:
    subject: int
    trial_index: int
    early_relevance: float
    late_relevance: float
    early_late_contrast: float
    correct: int


@dataclass(frozen=True)
class CspLdaRightHandEarlyLateContrastGeeRow:
    analysis: str
    model: str
    predictor: str
    coefficient: float
    standard_error: float
    odds_ratio: float
    odds_ratio_ci_low: float
    odds_ratio_ci_high: float
    p_value: float
    n_subjects: int
    n_right_hand_trials: int
    interpretation: str


COMPARISONS: tuple[
    TemporalComparison,
    ...,
] = (
    TemporalComparison(
        name="CSP+LDA CORRECT vs EEGNet CORRECT",
        slug="csp_lda_correct_vs_eegnet_correct",
        left_label="CSP correct",
        right_label="EEGNet correct",
        left_key=("csp", "correct"),
        right_key=("eegnet", "correct"),
        difference_label="EEGNet correct - CSP correct",
    ),
    TemporalComparison(
        name="EEGNet CORRECT vs EEGNet INCORRECT",
        slug="eegnet_correct_vs_eegnet_incorrect",
        left_label="EEGNet incorrect",
        right_label="EEGNet correct",
        left_key=("eegnet", "incorrect"),
        right_key=("eegnet", "correct"),
        difference_label="EEGNet correct - EEGNet incorrect",
    ),
    TemporalComparison(
        name="CSP+LDA CORRECT vs CSP+LDA INCORRECT",
        slug="csp_lda_correct_vs_csp_lda_incorrect",
        left_label="CSP incorrect",
        right_label="CSP correct",
        left_key=("csp", "incorrect"),
        right_key=("csp", "correct"),
        difference_label="CSP correct - CSP incorrect",
    ),
)


def run_temporal_statistical_analysis() -> tuple[
    list[SubjectTemporalProfile],
    list[TemporalStatisticRow],
]:
    """
    Run the subject-level temporal relevance statistical analysis.
    """
    profiles = collect_subject_temporal_profiles()
    rows = compute_temporal_statistics(
        profiles
    )

    save_temporal_profiles(
        profiles
    )

    save_temporal_statistics(
        rows
    )

    print_temporal_statistics(
        rows
    )

    return profiles, rows


def run_csp_lda_classwise_temporal_statistical_analysis() -> tuple[
    list[SubjectClassTemporalProfile],
    list[ClassTemporalStatisticRow],
]:
    """
    Run the class-wise CSP+LDA correct-vs-incorrect temporal analysis.
    """
    profiles = collect_csp_lda_classwise_temporal_profiles()
    rows = compute_csp_lda_classwise_temporal_statistics(
        profiles
    )

    save_csp_lda_classwise_temporal_statistics(
        rows
    )

    print_csp_lda_classwise_temporal_statistics(
        rows
    )

    return profiles, rows


def run_classwise_csp_lda_vs_eegnet_correct_temporal_analysis() -> tuple[
    list[SubjectModelClassTemporalProfile],
    list[ClassModelTemporalStatisticRow],
]:
    """
    Run class-wise CSP+LDA correct vs EEGNet correct temporal analysis.
    """
    profiles = collect_classwise_model_correct_temporal_profiles()
    rows = compute_classwise_csp_lda_vs_eegnet_correct_statistics(
        profiles
    )

    save_classwise_csp_lda_vs_eegnet_correct_statistics(
        rows
    )

    print_classwise_csp_lda_vs_eegnet_correct_statistics(
        rows
    )

    return profiles, rows


def run_left_hand_early_relevance_gee_analysis() -> tuple[
    list[LeftHandEarlyRelevanceTrial],
    LeftHandEarlyRelevanceGeeRow,
]:
    """
    Run a subject-clustered GEE model for Left-Hand early relevance.
    """
    trials = collect_left_hand_early_relevance_trials()
    row = fit_left_hand_early_relevance_gee(
        trials
    )

    save_left_hand_early_relevance_trials(
        trials
    )
    save_left_hand_early_relevance_gee_results(
        row
    )

    print_left_hand_early_relevance_gee_results(
        row
    )

    return trials, row


def run_csp_lda_right_hand_early_late_gee_analysis() -> tuple[
    list[CspLdaRightHandEarlyLateContrastGeeTrial],
    list[CspLdaRightHandEarlyLateContrastGeeRow],
]:
    """
    Run CSP+LDA Right-Hand Early/Late follow-up GEE models.
    """
    trials = collect_csp_lda_right_hand_early_late_gee_trials()
    rows = fit_csp_lda_right_hand_early_late_gee_models(
        trials
    )

    save_csp_lda_right_hand_early_late_gee_trials(
        trials
    )
    save_csp_lda_right_hand_early_late_gee_results(
        rows
    )

    print_csp_lda_right_hand_early_late_gee_results(
        rows
    )

    return trials, rows


def collect_subject_temporal_profiles() -> list[
    SubjectTemporalProfile
]:
    """
    Build one class-balanced temporal-window profile per subject.
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


def collect_csp_lda_classwise_temporal_profiles() -> list[
    SubjectClassTemporalProfile
]:
    """
    Build one normalized temporal-window profile per subject/class.
    """
    profiles = []

    for subject in SUBJECTS:
        profiles.extend(
            _load_csp_subject_classwise_profiles(
                subject
            )
        )

    return profiles


def collect_classwise_model_correct_temporal_profiles() -> list[
    SubjectModelClassTemporalProfile
]:
    """
    Build class-wise correct-trial profiles for CSP+LDA and EEGNet.
    """
    profiles = []

    for subject in SUBJECTS:
        profiles.extend(
            _load_csp_subject_model_classwise_profiles(
                subject,
                condition="correct",
            )
        )

        profiles.extend(
            _load_eegnet_subject_model_classwise_profiles(
                subject,
                condition="correct",
            )
        )

    return profiles


def collect_left_hand_early_relevance_trials() -> list[
    LeftHandEarlyRelevanceTrial
]:
    """
    Compute normalized early relevance for each Left-Hand trial.
    """
    trials = []

    for subject in SUBJECTS:
        trials.extend(
            _load_left_hand_early_relevance_trials_for_subject(
                subject
            )
        )

    return trials


def collect_csp_lda_right_hand_early_late_gee_trials() -> list[
    CspLdaRightHandEarlyLateContrastGeeTrial
]:
    """
    Compute normalized Early/Late contrast for each CSP+LDA Right-Hand trial.
    """
    trials = []

    for subject in SUBJECTS:
        trials.extend(
            _load_csp_lda_right_hand_early_late_gee_trials_for_subject(
                subject
            )
        )

    return trials


def compute_temporal_statistics(
    profiles: list[SubjectTemporalProfile],
) -> list[TemporalStatisticRow]:
    """
    Compute paired Wilcoxon tests and descriptives per window.
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

        for window_index, (window_name, start, end) in enumerate(
            TEMPORAL_WINDOWS
        ):
            left_values, right_values = _paired_window_values(
                profile_lookup=profile_lookup,
                comparison=comparison,
                window_index=window_index,
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
                TemporalStatisticRow(
                    comparison=comparison.name,
                    comparison_slug=comparison.slug,
                    temporal_window=window_name,
                    start_time=start,
                    end_time=end,
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
                TemporalStatisticRow(
                    comparison=row.comparison,
                    comparison_slug=row.comparison_slug,
                    temporal_window=row.temporal_window,
                    start_time=row.start_time,
                    end_time=row.end_time,
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


def compute_csp_lda_classwise_temporal_statistics(
    profiles: list[SubjectClassTemporalProfile],
) -> list[ClassTemporalStatisticRow]:
    """
    Compute paired Wilcoxon tests per class/window.

    FDR correction is applied across Early/Middle/Late separately
    within each class.
    """
    profile_lookup = {
        (
            profile.subject,
            profile.class_label,
            profile.condition,
        ): profile.values
        for profile in profiles
    }

    rows = []

    for class_label, class_name in zip(
        CLASS_LABELS,
        CLASS_NAMES,
        strict=True,
    ):
        class_rows = []
        raw_p_values = []
        display_name = _display_class_name(
            class_name
        )

        for window_index, (window_name, start, end) in enumerate(
            TEMPORAL_WINDOWS
        ):
            incorrect_values, correct_values = (
                _paired_class_window_values(
                    profile_lookup=profile_lookup,
                    class_label=class_label,
                    window_index=window_index,
                )
            )

            differences = (
                correct_values
                - incorrect_values
            )

            statistic, p_value = _paired_wilcoxon(
                differences
            )

            raw_p_values.append(
                p_value
            )

            class_rows.append(
                ClassTemporalStatisticRow(
                    class_label=class_label,
                    class_name=display_name,
                    temporal_window=window_name,
                    start_time=start,
                    end_time=end,
                    n=len(differences),
                    correct_mean=_nanmean(correct_values),
                    correct_std=_nanstd(correct_values),
                    incorrect_mean=_nanmean(incorrect_values),
                    incorrect_std=_nanstd(incorrect_values),
                    mean_difference=_nanmean(differences),
                    wilcoxon_statistic=statistic,
                    p_value=p_value,
                    p_value_fdr=np.nan,
                    significant_fdr=False,
                    direction=_classwise_direction_label(
                        _nanmean(differences)
                    ),
                )
            )

        corrected_p_values, significant = _fdr_correct(
            raw_p_values
        )

        for row, p_value_fdr, is_significant in zip(
            class_rows,
            corrected_p_values,
            significant,
            strict=True,
        ):
            rows.append(
                ClassTemporalStatisticRow(
                    class_label=row.class_label,
                    class_name=row.class_name,
                    temporal_window=row.temporal_window,
                    start_time=row.start_time,
                    end_time=row.end_time,
                    n=row.n,
                    correct_mean=row.correct_mean,
                    correct_std=row.correct_std,
                    incorrect_mean=row.incorrect_mean,
                    incorrect_std=row.incorrect_std,
                    mean_difference=row.mean_difference,
                    wilcoxon_statistic=row.wilcoxon_statistic,
                    p_value=row.p_value,
                    p_value_fdr=p_value_fdr,
                    significant_fdr=bool(
                        is_significant
                    ),
                    direction=row.direction,
                )
            )

    return rows


def compute_classwise_csp_lda_vs_eegnet_correct_statistics(
    profiles: list[SubjectModelClassTemporalProfile],
) -> list[ClassModelTemporalStatisticRow]:
    """
    Compute class-wise paired CSP+LDA correct vs EEGNet correct tests.

    FDR correction is applied across Early/Middle/Late separately
    within each class.
    """
    profile_lookup = {
        (
            profile.subject,
            profile.model,
            profile.class_label,
            profile.condition,
        ): profile.values
        for profile in profiles
    }

    rows = []

    for class_label, class_name in zip(
        CLASS_LABELS,
        CLASS_NAMES,
        strict=True,
    ):
        class_rows = []
        raw_p_values = []
        display_name = _display_class_name(
            class_name
        )

        for window_index, (window_name, start, end) in enumerate(
            TEMPORAL_WINDOWS
        ):
            csp_values, eegnet_values = (
                _paired_model_class_window_values(
                    profile_lookup=profile_lookup,
                    class_label=class_label,
                    window_index=window_index,
                    condition="correct",
                )
            )

            differences = (
                eegnet_values
                - csp_values
            )

            statistic, p_value = _paired_wilcoxon(
                differences
            )

            raw_p_values.append(
                p_value
            )

            class_rows.append(
                ClassModelTemporalStatisticRow(
                    class_label=class_label,
                    class_name=display_name,
                    temporal_window=window_name,
                    start_time=start,
                    end_time=end,
                    n=len(differences),
                    csp_correct_mean=_nanmean(csp_values),
                    csp_correct_std=_nanstd(csp_values),
                    eegnet_correct_mean=_nanmean(eegnet_values),
                    eegnet_correct_std=_nanstd(eegnet_values),
                    mean_difference=_nanmean(differences),
                    wilcoxon_statistic=statistic,
                    p_value=p_value,
                    p_value_fdr=np.nan,
                    significant_fdr=False,
                    direction=(
                        _classwise_model_direction_label(
                            _nanmean(differences)
                        )
                    ),
                )
            )

        corrected_p_values, significant = _fdr_correct(
            raw_p_values
        )

        for row, p_value_fdr, is_significant in zip(
            class_rows,
            corrected_p_values,
            significant,
            strict=True,
        ):
            rows.append(
                ClassModelTemporalStatisticRow(
                    class_label=row.class_label,
                    class_name=row.class_name,
                    temporal_window=row.temporal_window,
                    start_time=row.start_time,
                    end_time=row.end_time,
                    n=row.n,
                    csp_correct_mean=row.csp_correct_mean,
                    csp_correct_std=row.csp_correct_std,
                    eegnet_correct_mean=row.eegnet_correct_mean,
                    eegnet_correct_std=row.eegnet_correct_std,
                    mean_difference=row.mean_difference,
                    wilcoxon_statistic=row.wilcoxon_statistic,
                    p_value=row.p_value,
                    p_value_fdr=p_value_fdr,
                    significant_fdr=bool(
                        is_significant
                    ),
                    direction=row.direction,
                )
            )

    return rows


def fit_left_hand_early_relevance_gee(
    trials: list[LeftHandEarlyRelevanceTrial],
) -> LeftHandEarlyRelevanceGeeRow:
    """
    Fit correct ~ early_relevance using binomial GEE by subject.
    """
    finite_trials = [
        trial
        for trial in trials
        if np.isfinite(
            trial.early_relevance
        )
    ]

    if not finite_trials:
        raise ValueError(
            "No finite Left-Hand early-relevance trials are available."
        )

    correct = np.asarray(
        [
            trial.correct
            for trial in finite_trials
        ],
        dtype=np.float64,
    )

    if len(np.unique(correct)) < 2:
        raise ValueError(
            "GEE requires both correct and incorrect Left-Hand trials."
        )

    early_relevance = np.asarray(
        [
            trial.early_relevance
            for trial in finite_trials
        ],
        dtype=np.float64,
    )
    groups = np.asarray(
        [
            trial.subject
            for trial in finite_trials
        ]
    )

    predictors = sm.add_constant(
        early_relevance
    )

    model = sm.GEE(
        endog=correct,
        exog=predictors,
        groups=groups,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )

    result = model.fit()
    confidence_interval = np.asarray(
        result.conf_int()
    )

    coefficient = float(
        result.params[1]
    )
    standard_error = float(
        result.bse[1]
    )
    p_value = float(
        result.pvalues[1]
    )
    odds_ratio_ci_low = float(
        np.exp(
            confidence_interval[1, 0]
        )
    )
    odds_ratio_ci_high = float(
        np.exp(
            confidence_interval[1, 1]
        )
    )

    return LeftHandEarlyRelevanceGeeRow(
        coefficient=coefficient,
        standard_error=standard_error,
        odds_ratio=float(
            np.exp(
                coefficient
            )
        ),
        odds_ratio_ci_low=odds_ratio_ci_low,
        odds_ratio_ci_high=odds_ratio_ci_high,
        p_value=p_value,
        n_subjects=len(
            np.unique(
                groups
            )
        ),
        n_left_hand_trials=len(
            finite_trials
        ),
        interpretation=_left_hand_early_gee_interpretation(
            coefficient,
            p_value,
        ),
    )


def fit_csp_lda_right_hand_early_late_gee_models(
    trials: list[CspLdaRightHandEarlyLateContrastGeeTrial],
) -> list[CspLdaRightHandEarlyLateContrastGeeRow]:
    """
    Fit Right-Hand CSP+LDA contrast, Early-only, and Late-only GEE models.
    """
    return [
        _fit_csp_lda_right_hand_single_predictor_gee(
            trials=trials,
            model_label="Early-over-Late contrast GEE",
            predictor_name="early_late_contrast",
            interpretation_fn=(
                _csp_lda_right_hand_early_late_gee_interpretation
            ),
        ),
        _fit_csp_lda_right_hand_single_predictor_gee(
            trials=trials,
            model_label="Early-only GEE",
            predictor_name="early_relevance",
            interpretation_fn=(
                _csp_lda_right_hand_early_only_gee_interpretation
            ),
        ),
        _fit_csp_lda_right_hand_single_predictor_gee(
            trials=trials,
            model_label="Late-only GEE",
            predictor_name="late_relevance",
            interpretation_fn=(
                _csp_lda_right_hand_late_only_gee_interpretation
            ),
        ),
    ]


def _fit_csp_lda_right_hand_single_predictor_gee(
    trials: list[CspLdaRightHandEarlyLateContrastGeeTrial],
    model_label: str,
    predictor_name: str,
    interpretation_fn,
) -> CspLdaRightHandEarlyLateContrastGeeRow:
    """
    Fit one CSP+LDA Right-Hand binomial GEE model by subject.
    """
    finite_trials = [
        trial
        for trial in trials
        if np.isfinite(
            getattr(
                trial,
                predictor_name,
            )
        )
    ]

    if not finite_trials:
        raise ValueError(
            "No finite CSP+LDA Right-Hand Early/Late contrast trials "
            "are available."
        )

    correct = np.asarray(
        [
            trial.correct
            for trial in finite_trials
        ],
        dtype=np.float64,
    )

    if len(np.unique(correct)) < 2:
        raise ValueError(
            "GEE requires both correct and incorrect CSP+LDA "
            "Right-Hand trials."
        )

    predictor_values = np.asarray(
        [
            getattr(
                trial,
                predictor_name,
            )
            for trial in finite_trials
        ],
        dtype=np.float64,
    )
    groups = np.asarray(
        [
            trial.subject
            for trial in finite_trials
        ]
    )

    predictors = sm.add_constant(
        predictor_values
    )

    model = sm.GEE(
        endog=correct,
        exog=predictors,
        groups=groups,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )

    result = model.fit()
    confidence_interval = np.asarray(
        result.conf_int()
    )

    coefficient = float(
        result.params[1]
    )
    standard_error = float(
        result.bse[1]
    )
    p_value = float(
        result.pvalues[1]
    )
    odds_ratio_ci_low = float(
        np.exp(
            confidence_interval[1, 0]
        )
    )
    odds_ratio_ci_high = float(
        np.exp(
            confidence_interval[1, 1]
        )
    )

    return CspLdaRightHandEarlyLateContrastGeeRow(
        analysis=model_label,
        model=(
            "correct ~ "
            f"{predictor_name}"
        ),
        predictor=predictor_name,
        coefficient=coefficient,
        standard_error=standard_error,
        odds_ratio=float(
            np.exp(
                coefficient
            )
        ),
        odds_ratio_ci_low=odds_ratio_ci_low,
        odds_ratio_ci_high=odds_ratio_ci_high,
        p_value=p_value,
        n_subjects=len(
            np.unique(
                groups
            )
        ),
        n_right_hand_trials=len(
            finite_trials
        ),
        interpretation=(
            interpretation_fn(
                coefficient,
                p_value,
            )
        ),
    )


def save_temporal_profiles(
    profiles: list[SubjectTemporalProfile],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save subject-level normalized temporal-window profiles.
    """
    if output_file is None:
        output_file = get_temporal_statistical_profiles_path()

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
            "temporal_window",
            "start_time",
            "end_time",
            "normalized_mean_relevance",
        ])

        for profile in profiles:
            subject_name = get_subject_name(
                profile.subject
            )

            for (
                window_name,
                start,
                end,
            ), value in zip(
                TEMPORAL_WINDOWS,
                profile.values,
                strict=True,
            ):
                writer.writerow([
                    subject_name,
                    profile.model,
                    profile.condition,
                    window_name,
                    _format_float(start),
                    _format_float(end),
                    _format_float(value),
                ])

    return output_file


def save_temporal_statistics(
    rows: list[TemporalStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save temporal statistical test results.
    """
    if output_file is None:
        output_file = get_temporal_statistical_results_path()

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
                "temporal_window",
                "start_time",
                "end_time",
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
                "temporal_window": row.temporal_window,
                "start_time": _format_float(row.start_time),
                "end_time": _format_float(row.end_time),
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


def save_csp_lda_classwise_temporal_statistics(
    rows: list[ClassTemporalStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save class-wise CSP+LDA temporal statistical test results.
    """
    if output_file is None:
        output_file = get_temporal_classwise_statistical_results_path()

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
                "class_label",
                "class_name",
                "temporal_window",
                "start_time",
                "end_time",
                "n",
                "correct_mean",
                "correct_std",
                "incorrect_mean",
                "incorrect_std",
                "mean_difference",
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
                "comparison": (
                    "CSP+LDA correct vs incorrect"
                ),
                "class_label": row.class_label,
                "class_name": row.class_name,
                "temporal_window": row.temporal_window,
                "start_time": _format_float(row.start_time),
                "end_time": _format_float(row.end_time),
                "n": row.n,
                "correct_mean": _format_float(row.correct_mean),
                "correct_std": _format_float(row.correct_std),
                "incorrect_mean": _format_float(
                    row.incorrect_mean
                ),
                "incorrect_std": _format_float(
                    row.incorrect_std
                ),
                "mean_difference": _format_float(
                    row.mean_difference
                ),
                "wilcoxon_statistic": _format_float(
                    row.wilcoxon_statistic
                ),
                "p_value": _format_float(row.p_value),
                "p_value_fdr": _format_float(row.p_value_fdr),
                "significant_fdr": row.significant_fdr,
                "direction": row.direction,
            })

    return output_file


def save_classwise_csp_lda_vs_eegnet_correct_statistics(
    rows: list[ClassModelTemporalStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save class-wise CSP+LDA correct vs EEGNet correct test results.
    """
    if output_file is None:
        output_file = (
            get_temporal_classwise_csp_vs_eegnet_correct_results_path()
        )

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
                "class_label",
                "class_name",
                "temporal_window",
                "start_time",
                "end_time",
                "n",
                "csp_correct_mean",
                "csp_correct_std",
                "eegnet_correct_mean",
                "eegnet_correct_std",
                "mean_difference",
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
                "comparison": (
                    "CSP+LDA correct vs EEGNet correct"
                ),
                "class_label": row.class_label,
                "class_name": row.class_name,
                "temporal_window": row.temporal_window,
                "start_time": _format_float(row.start_time),
                "end_time": _format_float(row.end_time),
                "n": row.n,
                "csp_correct_mean": _format_float(
                    row.csp_correct_mean
                ),
                "csp_correct_std": _format_float(
                    row.csp_correct_std
                ),
                "eegnet_correct_mean": _format_float(
                    row.eegnet_correct_mean
                ),
                "eegnet_correct_std": _format_float(
                    row.eegnet_correct_std
                ),
                "mean_difference": _format_float(
                    row.mean_difference
                ),
                "wilcoxon_statistic": _format_float(
                    row.wilcoxon_statistic
                ),
                "p_value": _format_float(row.p_value),
                "p_value_fdr": _format_float(row.p_value_fdr),
                "significant_fdr": row.significant_fdr,
                "direction": row.direction,
            })

    return output_file


def save_left_hand_early_relevance_trials(
    trials: list[LeftHandEarlyRelevanceTrial],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save trial-level Left-Hand early relevance values.
    """
    if output_file is None:
        output_file = get_temporal_left_hand_early_gee_trials_path()

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
            "trial_index",
            "early_relevance",
            "correct",
        ])

        for trial in trials:
            writer.writerow([
                get_subject_name(
                    trial.subject
                ),
                trial.trial_index,
                _format_float(
                    trial.early_relevance
                ),
                trial.correct,
            ])

    return output_file


def save_left_hand_early_relevance_gee_results(
    row: LeftHandEarlyRelevanceGeeRow,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save the Left-Hand early-relevance GEE result.
    """
    if output_file is None:
        output_file = get_temporal_left_hand_early_gee_results_path()

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
                "model",
                "coefficient",
                "standard_error",
                "odds_ratio",
                "odds_ratio_ci_low",
                "odds_ratio_ci_high",
                "p_value",
                "n_subjects",
                "n_left_hand_trials",
                "interpretation",
            ],
        )

        writer.writeheader()
        writer.writerow({
            "model": "correct ~ early_relevance",
            "coefficient": _format_float(
                row.coefficient
            ),
            "standard_error": _format_float(
                row.standard_error
            ),
            "odds_ratio": _format_float(
                row.odds_ratio
            ),
            "odds_ratio_ci_low": _format_float(
                row.odds_ratio_ci_low
            ),
            "odds_ratio_ci_high": _format_float(
                row.odds_ratio_ci_high
            ),
            "p_value": _format_float(
                row.p_value
            ),
            "n_subjects": row.n_subjects,
            "n_left_hand_trials": row.n_left_hand_trials,
            "interpretation": row.interpretation,
        })

    return output_file


def save_csp_lda_right_hand_early_late_gee_trials(
    trials: list[CspLdaRightHandEarlyLateContrastGeeTrial],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save trial-level CSP+LDA Right-Hand Early/Late contrast values.
    """
    if output_file is None:
        output_file = (
            get_temporal_csp_lda_right_hand_early_late_gee_trials_path()
        )

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
            "trial_index",
            "early_relevance",
            "late_relevance",
            "early_late_contrast",
            "correct",
        ])

        for trial in trials:
            writer.writerow([
                get_subject_name(
                    trial.subject
                ),
                trial.trial_index,
                _format_float(
                    trial.early_relevance
                ),
                _format_float(
                    trial.late_relevance
                ),
                _format_float(
                    trial.early_late_contrast
                ),
                trial.correct,
            ])

    return output_file


def save_csp_lda_right_hand_early_late_gee_results(
    rows: list[CspLdaRightHandEarlyLateContrastGeeRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save the CSP+LDA Right-Hand Early/Late GEE results.
    """
    if output_file is None:
        output_file = (
            get_temporal_csp_lda_right_hand_early_late_gee_results_path()
        )

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
                "analysis",
                "model",
                "predictor",
                "coefficient",
                "standard_error",
                "odds_ratio",
                "odds_ratio_ci_low",
                "odds_ratio_ci_high",
                "p_value",
                "n_subjects",
                "n_right_hand_trials",
                "interpretation",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "analysis": row.analysis,
                "model": row.model,
                "predictor": row.predictor,
                "coefficient": _format_float(
                    row.coefficient
                ),
                "standard_error": _format_float(
                    row.standard_error
                ),
                "odds_ratio": _format_float(
                    row.odds_ratio
                ),
                "odds_ratio_ci_low": _format_float(
                    row.odds_ratio_ci_low
                ),
                "odds_ratio_ci_high": _format_float(
                    row.odds_ratio_ci_high
                ),
                "p_value": _format_float(
                    row.p_value
                ),
                "n_subjects": row.n_subjects,
                "n_right_hand_trials": row.n_right_hand_trials,
                "interpretation": row.interpretation,
            })

    return output_file


def print_temporal_statistics(
    rows: list[TemporalStatisticRow],
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
            f"{'Window':<10} "
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
                f"{row.temporal_window:<10} "
                f"{_mean_sd(row.mean_a, row.std_a):<24} "
                f"{_mean_sd(row.mean_b, row.std_b):<26} "
                f"{row.mean_difference:>8.4f} "
                f"{row.p_value:>10.4g} "
                f"{row.p_value_fdr:>10.4g}"
                f"{marker}"
            )


def print_csp_lda_classwise_temporal_statistics(
    rows: list[ClassTemporalStatisticRow],
) -> None:
    """
    Print a compact class-wise temporal summary.
    """
    print()
    print("=" * 70)
    print("CSP+LDA class-wise correct vs incorrect temporal relevance")
    print("=" * 70)
    print(
        f"{'Class':<12} "
        f"{'Window':<10} "
        f"{'n':>2} "
        f"{'Correct mean+/-SD':<22} "
        f"{'Incorrect mean+/-SD':<24} "
        f"{'Delta':>8} "
        f"{'p':>10} "
        f"{'p_FDR':>10}"
    )

    for row in rows:
        marker = (
            "*"
            if row.significant_fdr
            else ""
        )

        print(
            f"{row.class_name:<12} "
            f"{row.temporal_window:<10} "
            f"{row.n:>2} "
            f"{_mean_sd(row.correct_mean, row.correct_std):<22} "
            f"{_mean_sd(row.incorrect_mean, row.incorrect_std):<24} "
            f"{row.mean_difference:>8.4f} "
            f"{row.p_value:>10.4g} "
            f"{row.p_value_fdr:>10.4g}"
            f"{marker}"
        )


def print_classwise_csp_lda_vs_eegnet_correct_statistics(
    rows: list[ClassModelTemporalStatisticRow],
) -> None:
    """
    Print a compact class-wise CSP+LDA vs EEGNet correct summary.
    """
    print()
    print("=" * 70)
    print("Class-wise CSP+LDA correct vs EEGNet correct temporal relevance")
    print("=" * 70)
    print(
        f"{'Class':<12} "
        f"{'Window':<10} "
        f"{'n':>2} "
        f"{'CSP mean+/-SD':<20} "
        f"{'EEGNet mean+/-SD':<22} "
        f"{'Delta':>8} "
        f"{'p':>10} "
        f"{'p_FDR':>10}"
    )

    for row in rows:
        marker = (
            "*"
            if row.significant_fdr
            else ""
        )

        print(
            f"{row.class_name:<12} "
            f"{row.temporal_window:<10} "
            f"{row.n:>2} "
            f"{_mean_sd(row.csp_correct_mean, row.csp_correct_std):<20} "
            f"{_mean_sd(row.eegnet_correct_mean, row.eegnet_correct_std):<22} "
            f"{row.mean_difference:>8.4f} "
            f"{row.p_value:>10.4g} "
            f"{row.p_value_fdr:>10.4g}"
            f"{marker}"
        )


def print_left_hand_early_relevance_gee_results(
    row: LeftHandEarlyRelevanceGeeRow,
) -> None:
    """
    Print a concise GEE model summary.
    """
    print()
    print("=" * 70)
    print("CSP+LDA Left Hand early relevance GEE")
    print("=" * 70)
    print(
        "correct ~ early_relevance "
        f"(subjects={row.n_subjects}, "
        f"trials={row.n_left_hand_trials})"
    )
    print(
        f"coefficient={row.coefficient:.4g}, "
        f"SE={row.standard_error:.4g}, "
        f"OR={row.odds_ratio:.4g}, "
        "95% OR CI="
        f"[{row.odds_ratio_ci_low:.4g}, "
        f"{row.odds_ratio_ci_high:.4g}], "
        f"p={row.p_value:.4g}"
    )
    print(
        row.interpretation
    )


def print_csp_lda_right_hand_early_late_gee_results(
    rows: list[CspLdaRightHandEarlyLateContrastGeeRow],
) -> None:
    """
    Print concise CSP+LDA Right-Hand GEE model summaries.
    """
    for row in rows:
        print()
        print("=" * 70)
        print(f"CSP+LDA Right Hand {row.analysis}")
        print("=" * 70)
        print(
            f"{row.model} "
            f"(subjects={row.n_subjects}, "
            f"trials={row.n_right_hand_trials})"
        )
        print(
            f"coefficient={row.coefficient:.4g}, "
            f"SE={row.standard_error:.4g}, "
            f"OR={row.odds_ratio:.4g}, "
            "95% OR CI="
            f"[{row.odds_ratio_ci_low:.4g}, "
            f"{row.odds_ratio_ci_high:.4g}], "
            f"p={row.p_value:.4g}"
        )
        print(
            row.interpretation
        )


def _load_csp_subject_profiles(
    subject: int,
) -> list[SubjectTemporalProfile]:
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

    result = compute_trial_temporal_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
    )

    times = _create_times(
        result.values.shape[1]
    )

    profiles = []

    for condition, mask in (
        ("correct", result.correct_mask),
        ("incorrect", result.incorrect_mask),
    ):
        class_relevance, _ = aggregate_trial_temporal_relevance(
            result=result,
            mask=mask,
        )

        profiles.append(
            SubjectTemporalProfile(
                subject=subject,
                model="csp",
                condition=condition,
                values=_class_balanced_temporal_profile(
                    class_relevance=class_relevance,
                    times=times,
                ),
            )
        )

    return profiles


def _load_left_hand_early_relevance_trials_for_subject(
    subject: int,
) -> list[LeftHandEarlyRelevanceTrial]:
    """
    Compute trial-level early relevance for true Left-Hand trials.
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

    result = compute_trial_temporal_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
    )

    times = _create_times(
        result.values.shape[1]
    )

    left_hand_indices = np.flatnonzero(
        result.labels == LEFT_HAND_LABEL
    )

    trials = []

    for trial_index in left_hand_indices:
        early_relevance = (
            _single_trial_normalized_temporal_window_value(
                result.values[
                    trial_index
                ],
                times=times,
                start=0.5,
                end=1.5,
            )
        )

        trials.append(
            LeftHandEarlyRelevanceTrial(
                subject=subject,
                trial_index=int(
                    trial_index
                ),
                early_relevance=early_relevance,
                correct=int(
                    result.predictions[
                        trial_index
                    ]
                    == LEFT_HAND_LABEL
                ),
            )
        )

    return trials


def _load_csp_lda_right_hand_early_late_gee_trials_for_subject(
    subject: int,
) -> list[CspLdaRightHandEarlyLateContrastGeeTrial]:
    """
    Compute trial-level Early/Late contrast for true Right-Hand CSP+LDA trials.
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

    result = compute_trial_temporal_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
    )

    times = _create_times(
        result.values.shape[1]
    )

    right_hand_indices = np.flatnonzero(
        result.labels == RIGHT_HAND_LABEL
    )

    trials = []

    for trial_index in right_hand_indices:
        curve = result.values[
            trial_index
        ]
        early_relevance = (
            _single_trial_normalized_temporal_window_value(
                curve,
                times=times,
                start=0.5,
                end=1.5,
            )
        )
        late_relevance = (
            _single_trial_normalized_temporal_window_value(
                curve,
                times=times,
                start=2.5,
                end=4.0,
            )
        )

        trials.append(
            CspLdaRightHandEarlyLateContrastGeeTrial(
                subject=subject,
                trial_index=int(
                    trial_index
                ),
                early_relevance=early_relevance,
                late_relevance=late_relevance,
                early_late_contrast=(
                    early_relevance
                    - late_relevance
                ),
                correct=int(
                    result.predictions[
                        trial_index
                    ]
                    == RIGHT_HAND_LABEL
                ),
            )
        )

    return trials


def _load_csp_subject_classwise_profiles(
    subject: int,
) -> list[SubjectClassTemporalProfile]:
    """
    Compute CSP+LDA temporal profiles without averaging classes.
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

    result = compute_trial_temporal_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
    )

    times = _create_times(
        result.values.shape[1]
    )

    profiles = []

    for condition, mask in (
        ("correct", result.correct_mask),
        ("incorrect", result.incorrect_mask),
    ):
        class_relevance, _ = aggregate_trial_temporal_relevance(
            result=result,
            mask=mask,
        )

        relevance_by_class = dict(
            zip(
                np.unique(result.labels),
                class_relevance,
                strict=True,
            )
        )

        for class_label, class_name in zip(
            CLASS_LABELS,
            CLASS_NAMES,
            strict=True,
        ):
            class_curve = relevance_by_class.get(
                class_label,
                np.full(
                    result.values.shape[1],
                    np.nan,
                    dtype=np.float64,
                ),
            )

            profiles.append(
                SubjectClassTemporalProfile(
                    subject=subject,
                    class_label=class_label,
                    class_name=_display_class_name(
                        class_name
                    ),
                    condition=condition,
                    values=_single_class_temporal_profile(
                        class_curve,
                        times,
                    ),
                )
            )

    return profiles


def _load_csp_subject_model_classwise_profiles(
    subject: int,
    condition: TrialSelection,
) -> list[SubjectModelClassTemporalProfile]:
    """
    Compute class-wise CSP+LDA temporal profiles for one condition.
    """
    class_profiles = _load_csp_subject_classwise_profiles(
        subject
    )

    return [
        SubjectModelClassTemporalProfile(
            subject=profile.subject,
            model="csp",
            class_label=profile.class_label,
            class_name=profile.class_name,
            condition=profile.condition,
            values=profile.values,
        )
        for profile in class_profiles
        if profile.condition == condition
    ]


def _load_eegnet_subject_profiles(
    subject: int,
) -> list[SubjectTemporalProfile]:
    """
    Load saved time-domain SHAP values and summarize relevance.
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

    times = _create_times(
        result.values.shape[-1]
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

        temporal_relevance = compute_temporal_shap_relevance(
            class_relevance
        )

        profiles.append(
            SubjectTemporalProfile(
                subject=subject,
                model="eegnet",
                condition=condition,
                values=_class_balanced_temporal_profile(
                    class_relevance=(
                        _mapping_to_class_matrix(
                            temporal_relevance
                        )
                    ),
                    times=times,
                ),
            )
        )

    return profiles


def _load_eegnet_subject_model_classwise_profiles(
    subject: int,
    condition: TrialSelection,
) -> list[SubjectModelClassTemporalProfile]:
    """
    Load saved SHAP values and build class-wise EEGNet profiles.
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

    times = _create_times(
        result.values.shape[-1]
    )

    trial_mask = (
        result.correct_mask
        if condition == "correct"
        else result.incorrect_mask
    )

    class_relevance = _compute_class_shap_relevance(
        shap_values=result.values,
        labels=result.labels,
        trial_mask=trial_mask,
    )

    temporal_relevance = compute_temporal_shap_relevance(
        class_relevance
    )

    profiles = []

    for class_label, class_name in zip(
        CLASS_LABELS,
        CLASS_NAMES,
        strict=True,
    ):
        class_curve = temporal_relevance.get(
            class_label,
            np.full(
                len(times),
                np.nan,
                dtype=np.float64,
            ),
        )

        profiles.append(
            SubjectModelClassTemporalProfile(
                subject=subject,
                model="eegnet",
                class_label=class_label,
                class_name=_display_class_name(
                    class_name
                ),
                condition=condition,
                values=_single_class_temporal_profile(
                    class_curve,
                    times,
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
    Convert class-id keyed relevance to a class x time matrix.
    """
    n_times = _infer_n_times(
        relevance
    )

    matrix = np.full(
        (
            len(CLASS_LABELS),
            n_times,
        ),
        np.nan,
        dtype=np.float64,
    )

    for class_index, class_id in enumerate(
        CLASS_LABELS
    ):
        if class_id not in relevance:
            continue

        matrix[
            class_index
        ] = np.asarray(
            relevance[class_id],
            dtype=np.float64,
        )

    return matrix


def _infer_n_times(
    relevance: dict[int, np.ndarray],
) -> int:
    """
    Infer the number of time samples from the first available class.
    """
    for values in relevance.values():
        return len(
            values
        )

    raise ValueError(
        "No class relevance values are available."
    )


def _class_balanced_temporal_profile(
    class_relevance: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Area-normalize each class curve, window-average, then average classes.
    """
    class_relevance = np.asarray(
        class_relevance,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    class_window_values = np.full(
        (
            class_relevance.shape[0],
            len(TEMPORAL_WINDOWS),
        ),
        np.nan,
        dtype=np.float64,
    )

    normalization_mask = (
        (times >= NORMALIZATION_WINDOW[0])
        & (times <= NORMALIZATION_WINDOW[1])
    )

    for class_index in range(
        class_relevance.shape[0]
    ):
        curve = class_relevance[
            class_index
        ]

        if not np.any(
            np.isfinite(
                curve
            )
        ):
            continue

        denominator = np.trapezoid(
            curve[
                normalization_mask
            ],
            times[
                normalization_mask
            ],
        )

        if (
            not np.isfinite(
                denominator
            )
            or denominator <= 0
        ):
            continue

        normalized_curve = (
            curve / denominator
        )

        for window_index, (_, start, end) in enumerate(
            TEMPORAL_WINDOWS
        ):
            if window_index == len(TEMPORAL_WINDOWS) - 1:
                window_mask = (
                    (times >= start)
                    & (times <= end)
                )
            else:
                window_mask = (
                    (times >= start)
                    & (times < end)
                )

            class_window_values[
                class_index,
                window_index,
            ] = np.nanmean(
                normalized_curve[
                    window_mask
                ]
            )

    if np.all(
        np.isnan(
            class_window_values
        )
    ):
        return np.full(
            len(TEMPORAL_WINDOWS),
            np.nan,
            dtype=np.float64,
        )

    return np.nanmean(
        class_window_values,
        axis=0,
    )


def _single_class_temporal_profile(
    class_relevance: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    Area-normalize one class curve, then average temporal windows.
    """
    class_relevance = np.asarray(
        class_relevance,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    if not np.any(
        np.isfinite(
            class_relevance
        )
    ):
        return np.full(
            len(TEMPORAL_WINDOWS),
            np.nan,
            dtype=np.float64,
        )

    normalization_mask = (
        (times >= NORMALIZATION_WINDOW[0])
        & (times <= NORMALIZATION_WINDOW[1])
    )

    denominator = np.trapezoid(
        class_relevance[
            normalization_mask
        ],
        times[
            normalization_mask
        ],
    )

    if (
        not np.isfinite(
            denominator
        )
        or denominator <= 0
    ):
        return np.full(
            len(TEMPORAL_WINDOWS),
            np.nan,
            dtype=np.float64,
        )

    normalized_curve = (
        class_relevance
        / denominator
    )

    window_values = np.full(
        len(TEMPORAL_WINDOWS),
        np.nan,
        dtype=np.float64,
    )

    for window_index, (_, start, end) in enumerate(
        TEMPORAL_WINDOWS
    ):
        if window_index == len(TEMPORAL_WINDOWS) - 1:
            window_mask = (
                (times >= start)
                & (times <= end)
            )
        else:
            window_mask = (
                (times >= start)
                & (times < end)
            )

        window_values[
            window_index
        ] = np.nanmean(
            normalized_curve[
                window_mask
            ]
        )

    return window_values


def _single_trial_normalized_temporal_window_value(
    temporal_relevance: np.ndarray,
    times: np.ndarray,
    start: float,
    end: float,
) -> float:
    """
    Area-normalize one trial curve, then average one temporal window.
    """
    temporal_relevance = np.asarray(
        temporal_relevance,
        dtype=np.float64,
    )

    times = np.asarray(
        times,
        dtype=np.float64,
    )

    if not np.any(
        np.isfinite(
            temporal_relevance
        )
    ):
        return np.nan

    normalization_mask = (
        (times >= NORMALIZATION_WINDOW[0])
        & (times <= NORMALIZATION_WINDOW[1])
    )

    denominator = np.trapezoid(
        temporal_relevance[
            normalization_mask
        ],
        times[
            normalization_mask
        ],
    )

    if (
        not np.isfinite(
            denominator
        )
        or denominator <= 0
    ):
        return np.nan

    normalized_curve = (
        temporal_relevance
        / denominator
    )

    window_mask = (
        (times >= start)
        & (times < end)
    )

    return float(
        np.nanmean(
            normalized_curve[
                window_mask
            ]
        )
    )


def _create_times(
    n_times: int,
) -> np.ndarray:
    """
    Create the time axis for the classification epoch.
    """
    return (
        np.arange(
            n_times,
            dtype=np.float64,
        )
        / SFREQ
        + EPOCH_TMIN
    )


def _paired_window_values(
    profile_lookup: dict[
        tuple[int, ModelName, TrialSelection],
        np.ndarray,
    ],
    comparison: TemporalComparison,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return paired left/right values for one comparison and window.
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
            window_index
        ]

        right_value = right_profile[
            window_index
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


def _paired_class_window_values(
    profile_lookup: dict[
        tuple[int, int, TrialSelection],
        np.ndarray,
    ],
    class_label: int,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return paired incorrect/correct values for one class/window.
    """
    incorrect_values = []
    correct_values = []

    for subject in SUBJECTS:
        incorrect_profile = profile_lookup.get((
            subject,
            class_label,
            "incorrect",
        ))

        correct_profile = profile_lookup.get((
            subject,
            class_label,
            "correct",
        ))

        if (
            incorrect_profile is None
            or correct_profile is None
        ):
            continue

        incorrect_value = incorrect_profile[
            window_index
        ]
        correct_value = correct_profile[
            window_index
        ]

        if not (
            np.isfinite(incorrect_value)
            and np.isfinite(correct_value)
        ):
            continue

        incorrect_values.append(
            incorrect_value
        )
        correct_values.append(
            correct_value
        )

    return (
        np.asarray(
            incorrect_values,
            dtype=np.float64,
        ),
        np.asarray(
            correct_values,
            dtype=np.float64,
        ),
    )


def _paired_model_class_window_values(
    profile_lookup: dict[
        tuple[int, ModelName, int, TrialSelection],
        np.ndarray,
    ],
    class_label: int,
    window_index: int,
    condition: TrialSelection,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return paired CSP+LDA/EEGNet values for one class/window.
    """
    csp_values = []
    eegnet_values = []

    for subject in SUBJECTS:
        csp_profile = profile_lookup.get((
            subject,
            "csp",
            class_label,
            condition,
        ))

        eegnet_profile = profile_lookup.get((
            subject,
            "eegnet",
            class_label,
            condition,
        ))

        if (
            csp_profile is None
            or eegnet_profile is None
        ):
            continue

        csp_value = csp_profile[
            window_index
        ]
        eegnet_value = eegnet_profile[
            window_index
        ]

        if not (
            np.isfinite(csp_value)
            and np.isfinite(eegnet_value)
        ):
            continue

        csp_values.append(
            csp_value
        )
        eegnet_values.append(
            eegnet_value
        )

    return (
        np.asarray(
            csp_values,
            dtype=np.float64,
        ),
        np.asarray(
            eegnet_values,
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
    comparison: TemporalComparison,
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


def _classwise_direction_label(
    mean_difference: float,
) -> str:
    """
    Build a readable direction for CSP correct - CSP incorrect.
    """
    if not np.isfinite(
        mean_difference
    ):
        return "not available"

    if mean_difference > 0:
        return "CSP correct > CSP incorrect"

    if mean_difference < 0:
        return "CSP correct < CSP incorrect"

    return "CSP correct = CSP incorrect"


def _classwise_model_direction_label(
    mean_difference: float,
) -> str:
    """
    Build a readable direction for EEGNet correct - CSP correct.
    """
    if not np.isfinite(
        mean_difference
    ):
        return "not available"

    if mean_difference > 0:
        return "EEGNet correct > CSP correct"

    if mean_difference < 0:
        return "EEGNet correct < CSP correct"

    return "EEGNet correct = CSP correct"


def _left_hand_early_gee_interpretation(
    coefficient: float,
    p_value: float,
) -> str:
    """
    Interpret the early-relevance GEE coefficient.
    """
    if (
        np.isfinite(coefficient)
        and np.isfinite(p_value)
        and coefficient > 0
        and p_value < ALPHA
    ):
        return (
            "Greater early temporal relevance is associated with "
            "a higher probability of correct Left Hand classification."
        )

    return (
        "There is insufficient evidence that early temporal relevance "
        "is associated with correct Left Hand classification."
    )


def _csp_lda_right_hand_early_late_gee_interpretation(
    coefficient: float,
    p_value: float,
) -> str:
    """
    Interpret the CSP+LDA Right-Hand Early/Late contrast GEE coefficient.
    """
    if (
        np.isfinite(coefficient)
        and np.isfinite(p_value)
        and coefficient > 0
        and p_value < ALPHA
    ):
        return (
            "CSP+LDA Right Hand trials with relatively higher Early "
            "and lower Late relevance are more likely to be classified "
            "correctly."
        )

    return (
        "There is insufficient evidence that this temporal pattern is "
        "associated with CSP+LDA Right Hand correctness."
    )


def _csp_lda_right_hand_early_only_gee_interpretation(
    coefficient: float,
    p_value: float,
) -> str:
    """
    Interpret the CSP+LDA Right-Hand Early-only GEE coefficient.
    """
    if (
        np.isfinite(coefficient)
        and np.isfinite(p_value)
        and coefficient > 0
        and p_value < ALPHA
    ):
        return (
            "Greater Early relevance is associated with a higher "
            "probability of correct CSP+LDA Right Hand classification."
        )

    return (
        "There is insufficient evidence that Early relevance is "
        "associated with correct CSP+LDA Right Hand classification."
    )


def _csp_lda_right_hand_late_only_gee_interpretation(
    coefficient: float,
    p_value: float,
) -> str:
    """
    Interpret the CSP+LDA Right-Hand Late-only GEE coefficient.
    """
    if (
        np.isfinite(coefficient)
        and np.isfinite(p_value)
        and coefficient < 0
        and p_value < ALPHA
    ):
        return (
            "Greater Late relevance is associated with a lower "
            "probability of correct CSP+LDA Right Hand classification."
        )

    return (
        "There is insufficient evidence that Late relevance is "
        "associated with correct CSP+LDA Right Hand classification."
    )


def _display_class_name(
    class_name: str,
) -> str:
    return class_name.title()


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
