import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.csp_pattern_analysis.temporal_relevance import (
    aggregate_trial_temporal_relevance,
    compute_trial_temporal_relevance,
)
from src.analysis.temporal_statistical_analysis import (
    SFREQ,
    SUBJECTS,
    TEMPORAL_WINDOWS,
    _class_balanced_temporal_profile,
    _class_window_profiles,
    _create_times,
    _fdr_correct,
    _nanmean,
    _nanmedian,
    _nanstd,
    _paired_wilcoxon,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_NAMES
from src.utils.config import BASE_SEED, EPOCH_TMIN, N_FOLDS
from src.utils.paths import (
    EARLY_WEIGHTED_CSP_RESULTS_DIR,
    get_early_weighted_csp_fold_model_path,
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_subject_name,
)
from src.visualization.common import save_figure


MODEL_BASELINE = "Baseline CSP+LDA"
MODEL_WEIGHTED = "Early-weighted CSP+LDA"
MODEL_ORDER = (MODEL_BASELINE, MODEL_WEIGHTED)
SELECTION_MAIN = "model_correct"
SELECTION_SHARED = "shared_correct"

EARLY_START = 0.5
EARLY_END = 1.5
EARLY_WEIGHT = 2.0

OUTPUT_DIR = EARLY_WEIGHTED_CSP_RESULTS_DIR / "metrics"
STATISTICS_DIR = EARLY_WEIGHTED_CSP_RESULTS_DIR / "statistics"
FIGURE_DIR = EARLY_WEIGHTED_CSP_RESULTS_DIR / "figures"

SUBJECT_VALUES_PATH = (
    OUTPUT_DIR
    / "csp_baseline_vs_early_weighted_temporal_subject_values.csv"
)
TRIAL_COUNTS_PATH = (
    OUTPUT_DIR
    / "csp_baseline_vs_early_weighted_temporal_trial_counts.csv"
)
OVERALL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "csp_baseline_vs_early_weighted_temporal_overall.csv"
)
CLASSWISE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "csp_baseline_vs_early_weighted_temporal_by_class.csv"
)
STATISTICS_PATH = (
    STATISTICS_DIR
    / "csp_baseline_vs_early_weighted_temporal_statistics.csv"
)
SHARED_OVERALL_SUMMARY_PATH = (
    OUTPUT_DIR
    / "csp_baseline_vs_early_weighted_shared_correct_temporal_overall.csv"
)
SHARED_CLASSWISE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "csp_baseline_vs_early_weighted_shared_correct_temporal_by_class.csv"
)
SHARED_STATISTICS_PATH = (
    STATISTICS_DIR
    / "csp_baseline_vs_early_weighted_shared_correct_temporal_statistics.csv"
)
OVERALL_FIGURE_PATH = (
    FIGURE_DIR
    / "csp_baseline_vs_early_weighted_all_mean_correct_temporal_relevance.png"
)
CLASSWISE_FIGURE_PATH = (
    FIGURE_DIR
    / "csp_baseline_vs_early_weighted_all_mean_correct_temporal_relevance_by_class.png"
)
SHARED_CLASSWISE_FIGURE_PATH = (
    FIGURE_DIR
    / "csp_baseline_vs_early_weighted_shared_correct_temporal_relevance_by_class.png"
)


@dataclass(frozen=True)
class SubjectProfile:
    subject: int
    model: str
    selection: str
    class_values: np.ndarray
    class_counts: np.ndarray

    @property
    def values(self) -> np.ndarray:
        return _class_balanced_temporal_profile(
            self.class_values
        )


@dataclass(frozen=True)
class SummaryRow:
    scope: str
    class_name: str
    model: str
    early: float
    middle: float
    late: float


@dataclass(frozen=True)
class ChangeRow:
    scope: str
    class_name: str
    early: float
    middle: float
    late: float


@dataclass(frozen=True)
class StatisticRow:
    scope: str
    class_name: str
    temporal_window: str
    start_time: float
    end_time: float
    n: int
    baseline_mean: float
    baseline_std: float
    baseline_median: float
    weighted_mean: float
    weighted_std: float
    weighted_median: float
    mean_difference: float
    median_difference: float
    wilcoxon_statistic: float
    p_value: float
    p_value_fdr: float
    significant_fdr: bool


def main() -> None:
    profiles = collect_profiles()
    main_summary_rows, main_change_rows = summarize_profiles(
        profiles,
        selection=SELECTION_MAIN,
    )
    main_statistic_rows = compute_statistics(
        profiles,
        selection=SELECTION_MAIN,
    )
    shared_summary_rows, shared_change_rows = summarize_profiles(
        profiles,
        selection=SELECTION_SHARED,
    )
    shared_statistic_rows = compute_statistics(
        profiles,
        selection=SELECTION_SHARED,
    )

    save_subject_values(
        profiles,
        SUBJECT_VALUES_PATH,
    )
    save_trial_counts(
        profiles,
        TRIAL_COUNTS_PATH,
    )
    save_summary_rows(
        main_summary_rows,
        main_change_rows,
        OVERALL_SUMMARY_PATH,
        scope="overall",
    )
    save_summary_rows(
        main_summary_rows,
        main_change_rows,
        CLASSWISE_SUMMARY_PATH,
        scope="classwise",
    )
    save_statistics(
        main_statistic_rows,
        STATISTICS_PATH,
    )
    save_summary_rows(
        shared_summary_rows,
        shared_change_rows,
        SHARED_OVERALL_SUMMARY_PATH,
        scope="overall",
    )
    save_summary_rows(
        shared_summary_rows,
        shared_change_rows,
        SHARED_CLASSWISE_SUMMARY_PATH,
        scope="classwise",
    )
    save_statistics(
        shared_statistic_rows,
        SHARED_STATISTICS_PATH,
    )
    plot_overall(
        profiles=[
            profile
            for profile in profiles
            if profile.selection == SELECTION_MAIN
        ],
        statistic_rows=main_statistic_rows,
        output_path=OVERALL_FIGURE_PATH,
    )
    plot_classwise(
        profiles=[
            profile
            for profile in profiles
            if profile.selection == SELECTION_MAIN
        ],
        statistic_rows=main_statistic_rows,
        output_path=CLASSWISE_FIGURE_PATH,
    )
    plot_classwise(
        profiles=[
            profile
            for profile in profiles
            if profile.selection == SELECTION_SHARED
        ],
        statistic_rows=shared_statistic_rows,
        output_path=SHARED_CLASSWISE_FIGURE_PATH,
        title="CSP+LDA temporal relevance by class (shared correct trials)",
    )
    print_summary(
        title="Main analysis: model-specific correctly classified trials",
        summary_rows=main_summary_rows,
        change_rows=main_change_rows,
        statistic_rows=main_statistic_rows,
    )
    print_summary(
        title="Robustness check: shared correctly classified trials",
        summary_rows=shared_summary_rows,
        change_rows=shared_change_rows,
        statistic_rows=shared_statistic_rows,
    )
    print_trial_counts(
        profiles
    )
    print()
    print(f"Saved subject values: {SUBJECT_VALUES_PATH}")
    print(f"Saved trial counts: {TRIAL_COUNTS_PATH}")
    print(f"Saved overall summary: {OVERALL_SUMMARY_PATH}")
    print(f"Saved class-wise summary: {CLASSWISE_SUMMARY_PATH}")
    print(f"Saved statistics: {STATISTICS_PATH}")
    print(f"Saved shared-correct overall summary: {SHARED_OVERALL_SUMMARY_PATH}")
    print(f"Saved shared-correct class-wise summary: {SHARED_CLASSWISE_SUMMARY_PATH}")
    print(f"Saved shared-correct statistics: {SHARED_STATISTICS_PATH}")
    print(f"Saved overall figure: {OVERALL_FIGURE_PATH}")
    print(f"Saved class-wise figure: {CLASSWISE_FIGURE_PATH}")
    print(f"Saved shared-correct class-wise figure: {SHARED_CLASSWISE_FIGURE_PATH}")


def collect_profiles() -> list[SubjectProfile]:
    profiles = []

    for subject in SUBJECTS:
        subject_name = get_subject_name(subject)
        print(f"Processing {subject_name}...")

        data = get_data_for_subject(
            subject
        )

        if data is None:
            raise RuntimeError(
                f"Could not load data for {subject_name}."
            )

        _, _, x_eval, y_eval = data

        baseline_result = _compute_trial_result(
            csps_ldas=_load_baseline_models(subject),
            x_eval=x_eval,
            y_eval=y_eval,
        )

        weighted_x_eval = apply_temporal_scaling(
            x_eval,
            make_temporal_scaling(x_eval.shape[-1]),
        )
        weighted_result = _compute_trial_result(
            csps_ldas=_load_weighted_models(subject),
            x_eval=weighted_x_eval,
            y_eval=y_eval,
        )

        profiles.append(
            _make_subject_profile(
                subject=subject,
                model=MODEL_BASELINE,
                selection=SELECTION_MAIN,
                result=baseline_result,
                mask=baseline_result.correct_mask,
            )
        )
        profiles.append(
            _make_subject_profile(
                subject=subject,
                model=MODEL_WEIGHTED,
                selection=SELECTION_MAIN,
                result=weighted_result,
                mask=weighted_result.correct_mask,
            )
        )

        shared_correct_mask = (
            baseline_result.correct_mask
            & weighted_result.correct_mask
        )
        profiles.append(
            _make_subject_profile(
                subject=subject,
                model=MODEL_BASELINE,
                selection=SELECTION_SHARED,
                result=baseline_result,
                mask=shared_correct_mask,
            )
        )
        profiles.append(
            _make_subject_profile(
                subject=subject,
                model=MODEL_WEIGHTED,
                selection=SELECTION_SHARED,
                result=weighted_result,
                mask=shared_correct_mask,
            )
        )

    return profiles


def _compute_trial_result(
    csps_ldas: list,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
):
    csps = [
        csp
        for csp, _ in csps_ldas
    ]
    ldas = [
        lda
        for _, lda in csps_ldas
    ]

    return compute_trial_temporal_relevance(
        csps=csps,
        ldas=ldas,
        data=x_eval,
        labels=y_eval,
    )


def _make_subject_profile(
    subject: int,
    model: str,
    selection: str,
    result,
    mask: np.ndarray,
) -> SubjectProfile:
    class_relevance, _ = aggregate_trial_temporal_relevance(
        result=result,
        mask=mask,
    )
    times = _create_times(
        result.values.shape[1]
    )

    return SubjectProfile(
        subject=subject,
        model=model,
        selection=selection,
        class_values=_class_window_profiles(
            class_relevance=class_relevance,
            times=times,
        ),
        class_counts=_class_counts(
            labels=result.labels,
            mask=mask,
        ),
    )


def _load_baseline_models(
    subject: int,
) -> list:
    models = []

    for fold in _get_baseline_fold_numbers(
        subject
    ):
        models.append((
            joblib.load(
                get_csp_fold_model_path(subject, fold)
            ),
            joblib.load(
                get_lda_fold_model_path(subject, fold)
            ),
        ))

    return models


def _load_weighted_models(
    subject: int,
) -> list:
    models = []

    for fold in range(1, N_FOLDS + 1):
        csp_path = get_early_weighted_csp_fold_model_path(
            subject,
            fold,
            "csp",
        )
        lda_path = get_early_weighted_csp_fold_model_path(
            subject,
            fold,
            "lda",
        )

        if not (
            csp_path.exists()
            and lda_path.exists()
        ):
            raise FileNotFoundError(
                "Missing weighted CSP+LDA model files: "
                f"{csp_path} / {lda_path}"
            )

        models.append((
            joblib.load(csp_path),
            joblib.load(lda_path),
        ))

    return models


def _get_baseline_fold_numbers(
    subject: int,
) -> list[int]:
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
        "Could not find all baseline CSP+LDA fold models for "
        f"{get_subject_name(subject)}."
    )


def make_temporal_scaling(
    n_times: int,
) -> np.ndarray:
    times = (
        EPOCH_TMIN
        + np.arange(
            n_times,
            dtype=np.float64,
        )
        / SFREQ
    )
    early_mask = (
        (times >= EARLY_START)
        & (times < EARLY_END)
    )

    if not np.any(
        early_mask
    ):
        raise RuntimeError(
            "The Early temporal mask is empty."
        )

    sample_weights = np.ones(
        n_times,
        dtype=np.float64,
    )
    sample_weights[early_mask] = EARLY_WEIGHT

    return np.sqrt(
        sample_weights
    )


def apply_temporal_scaling(
    x: np.ndarray,
    temporal_scaling: np.ndarray,
) -> np.ndarray:
    weighted = x * temporal_scaling[np.newaxis, np.newaxis, :]

    if weighted.shape != x.shape:
        raise RuntimeError(
            "Temporal weighting changed data shape."
        )

    return weighted


def summarize_profiles(
    profiles: list[SubjectProfile],
    selection: str,
) -> tuple[
    list[SummaryRow],
    list[ChangeRow],
]:
    summary_rows = []
    change_rows = []

    for model in MODEL_ORDER:
        model_profiles = [
            profile
            for profile in profiles
            if (
                profile.model == model
                and profile.selection == selection
            )
        ]
        values = np.stack([
            profile.values
            for profile in model_profiles
        ])
        summary_rows.append(
            SummaryRow(
                scope="overall",
                class_name="all_classes_balanced",
                model=model,
                early=_nanmean(values[:, 0]),
                middle=_nanmean(values[:, 1]),
                late=_nanmean(values[:, 2]),
            )
        )

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):
            class_values = np.stack([
                profile.class_values[class_index]
                for profile in model_profiles
            ])
            summary_rows.append(
                SummaryRow(
                    scope="classwise",
                    class_name=class_name,
                    model=model,
                    early=_nanmean(class_values[:, 0]),
                    middle=_nanmean(class_values[:, 1]),
                    late=_nanmean(class_values[:, 2]),
                )
            )

    for scope, class_name in _summary_scopes():
        baseline = _find_summary(
            summary_rows,
            scope,
            class_name,
            MODEL_BASELINE,
        )
        weighted = _find_summary(
            summary_rows,
            scope,
            class_name,
            MODEL_WEIGHTED,
        )
        change_rows.append(
            ChangeRow(
                scope=scope,
                class_name=class_name,
                early=weighted.early - baseline.early,
                middle=weighted.middle - baseline.middle,
                late=weighted.late - baseline.late,
            )
        )

    return summary_rows, change_rows


def compute_statistics(
    profiles: list[SubjectProfile],
    selection: str,
) -> list[StatisticRow]:
    selected_profiles = [
        profile
        for profile in profiles
        if profile.selection == selection
    ]
    rows = []

    rows.extend(
        _compute_scope_statistics(
            profiles=selected_profiles,
            scope="overall",
            class_name="all_classes_balanced",
            class_index=None,
        )
    )

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        rows.extend(
            _compute_scope_statistics(
                profiles=selected_profiles,
                scope="classwise",
                class_name=class_name,
                class_index=class_index,
            )
        )

    return rows


def _compute_scope_statistics(
    profiles: list[SubjectProfile],
    scope: str,
    class_name: str,
    class_index: int | None,
) -> list[StatisticRow]:
    rows = []
    raw_p_values = []

    for window_index, (window_name, start, end) in enumerate(
        TEMPORAL_WINDOWS
    ):
        baseline_values, weighted_values = _paired_values(
            profiles=profiles,
            class_index=class_index,
            window_index=window_index,
        )
        differences = (
            weighted_values
            - baseline_values
        )
        statistic, p_value = _paired_wilcoxon(
            differences
        )
        row = StatisticRow(
            scope=scope,
            class_name=class_name,
            temporal_window=window_name,
            start_time=start,
            end_time=end,
            n=len(differences),
            baseline_mean=_nanmean(baseline_values),
            baseline_std=_nanstd(baseline_values),
            baseline_median=_nanmedian(baseline_values),
            weighted_mean=_nanmean(weighted_values),
            weighted_std=_nanstd(weighted_values),
            weighted_median=_nanmedian(weighted_values),
            mean_difference=_nanmean(differences),
            median_difference=_nanmedian(differences),
            wilcoxon_statistic=statistic,
            p_value=p_value,
            p_value_fdr=np.nan,
            significant_fdr=False,
        )
        rows.append(
            row
        )
        raw_p_values.append(
            p_value
        )

    corrected_p_values, significant = _fdr_correct(
        raw_p_values
    )

    return [
        StatisticRow(
            scope=row.scope,
            class_name=row.class_name,
            temporal_window=row.temporal_window,
            start_time=row.start_time,
            end_time=row.end_time,
            n=row.n,
            baseline_mean=row.baseline_mean,
            baseline_std=row.baseline_std,
            baseline_median=row.baseline_median,
            weighted_mean=row.weighted_mean,
            weighted_std=row.weighted_std,
            weighted_median=row.weighted_median,
            mean_difference=row.mean_difference,
            median_difference=row.median_difference,
            wilcoxon_statistic=row.wilcoxon_statistic,
            p_value=row.p_value,
            p_value_fdr=float(q_value),
            significant_fdr=bool(is_significant),
        )
        for row, q_value, is_significant in zip(
            rows,
            corrected_p_values,
            significant,
            strict=True,
        )
    ]


def _paired_values(
    profiles: list[SubjectProfile],
    class_index: int | None,
    window_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    by_subject_model = {
        (profile.subject, profile.model): profile
        for profile in profiles
    }
    baseline_values = []
    weighted_values = []

    for subject in SUBJECTS:
        baseline = by_subject_model.get((
            subject,
            MODEL_BASELINE,
        ))
        weighted = by_subject_model.get((
            subject,
            MODEL_WEIGHTED,
        ))

        if baseline is None or weighted is None:
            continue

        if class_index is None:
            baseline_value = baseline.values[window_index]
            weighted_value = weighted.values[window_index]
        else:
            baseline_value = baseline.class_values[
                class_index,
                window_index,
            ]
            weighted_value = weighted.class_values[
                class_index,
                window_index,
            ]

        if not (
            np.isfinite(baseline_value)
            and np.isfinite(weighted_value)
        ):
            continue

        baseline_values.append(
            baseline_value
        )
        weighted_values.append(
            weighted_value
        )

    return (
        np.asarray(
            baseline_values,
            dtype=np.float64,
        ),
        np.asarray(
            weighted_values,
            dtype=np.float64,
        ),
    )


def save_subject_values(
    profiles: list[SubjectProfile],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "selection",
                "scope",
                "class",
                "model",
                "early",
                "middle",
                "late",
            ],
        )
        writer.writeheader()

        for profile in profiles:
            writer.writerow({
                "subject": get_subject_name(profile.subject),
                "selection": profile.selection,
                "scope": "overall",
                "class": "all_classes_balanced",
                "model": profile.model,
                "early": _format_float(profile.values[0]),
                "middle": _format_float(profile.values[1]),
                "late": _format_float(profile.values[2]),
            })

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                values = profile.class_values[
                    class_index
                ]
                writer.writerow({
                    "subject": get_subject_name(profile.subject),
                    "selection": profile.selection,
                    "scope": "classwise",
                    "class": class_name,
                    "model": profile.model,
                    "early": _format_float(values[0]),
                    "middle": _format_float(values[1]),
                    "late": _format_float(values[2]),
                })


def save_trial_counts(
    profiles: list[SubjectProfile],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "selection",
                "model",
                "class",
                "n_trials",
            ],
        )
        writer.writeheader()

        for profile in profiles:
            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                writer.writerow({
                    "subject": get_subject_name(profile.subject),
                    "selection": profile.selection,
                    "model": profile.model,
                    "class": class_name,
                    "n_trials": int(
                        profile.class_counts[class_index]
                    ),
                })


def save_summary_rows(
    summary_rows: list[SummaryRow],
    change_rows: list[ChangeRow],
    output_path: Path,
    scope: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class",
                "model",
                "early",
                "middle",
                "late",
            ],
        )
        writer.writeheader()

        for row in summary_rows:
            if row.scope != scope:
                continue
            writer.writerow({
                "class": row.class_name,
                "model": row.model,
                "early": _format_float(row.early),
                "middle": _format_float(row.middle),
                "late": _format_float(row.late),
            })

        for row in change_rows:
            if row.scope != scope:
                continue
            writer.writerow({
                "class": row.class_name,
                "model": "Weighted - Baseline",
                "early": _format_float(row.early),
                "middle": _format_float(row.middle),
                "late": _format_float(row.late),
            })


def save_statistics(
    rows: list[StatisticRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scope",
                "class",
                "temporal_window",
                "n",
                "baseline_mean",
                "baseline_std",
                "baseline_median",
                "weighted_mean",
                "weighted_std",
                "weighted_median",
                "mean_difference",
                "median_difference",
                "wilcoxon_statistic",
                "p_value",
                "p_value_fdr",
                "significant_fdr",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "scope": row.scope,
                "class": row.class_name,
                "temporal_window": row.temporal_window,
                "n": row.n,
                "baseline_mean": _format_float(row.baseline_mean),
                "baseline_std": _format_float(row.baseline_std),
                "baseline_median": _format_float(row.baseline_median),
                "weighted_mean": _format_float(row.weighted_mean),
                "weighted_std": _format_float(row.weighted_std),
                "weighted_median": _format_float(row.weighted_median),
                "mean_difference": _format_float(row.mean_difference),
                "median_difference": _format_float(row.median_difference),
                "wilcoxon_statistic": _format_float(
                    row.wilcoxon_statistic
                ),
                "p_value": _format_float(row.p_value),
                "p_value_fdr": _format_float(row.p_value_fdr),
                "significant_fdr": row.significant_fdr,
            })


def plot_overall(
    profiles: list[SubjectProfile],
    statistic_rows: list[StatisticRow],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 5),
        constrained_layout=True,
    )
    _plot_profile_axis(
        axis=axis,
        profiles=profiles,
        statistic_rows=[
            row
            for row in statistic_rows
            if row.scope == "overall"
        ],
        class_index=None,
    )
    axis.set_title(
        "CSP+LDA temporal relevance: baseline vs early-weighted"
    )
    axis.legend()
    save_figure(
        figure,
        output_path,
    )
    plt.close(
        figure
    )


def plot_classwise(
    profiles: list[SubjectProfile],
    statistic_rows: list[StatisticRow],
    output_path: Path,
    title: str = "CSP+LDA temporal relevance by class",
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharey=True,
        constrained_layout=True,
    )

    for axis, class_index, class_name in zip(
        axes.ravel(),
        range(len(CLASS_NAMES)),
        CLASS_NAMES,
        strict=True,
    ):
        _plot_profile_axis(
            axis=axis,
            profiles=profiles,
            statistic_rows=[
                row
                for row in statistic_rows
                if (
                    row.scope == "classwise"
                    and row.class_name == class_name
                )
            ],
            class_index=class_index,
        )
        axis.set_title(
            class_name
        )

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncols=2,
    )
    figure.suptitle(
        title,
        y=1.03,
    )
    save_figure(
        figure,
        output_path,
    )
    plt.close(
        figure
    )


def _plot_profile_axis(
    axis,
    profiles: list[SubjectProfile],
    statistic_rows: list[StatisticRow],
    class_index: int | None,
) -> None:
    x_positions = np.arange(
        len(TEMPORAL_WINDOWS)
    )
    offset = 0.12

    for model, shift in (
        (MODEL_BASELINE, -offset),
        (MODEL_WEIGHTED, offset),
    ):
        values = _plot_values(
            profiles,
            model,
            class_index,
        )
        means = np.nanmean(
            values,
            axis=0,
        )
        stds = np.nanstd(
            values,
            axis=0,
            ddof=1,
        )
        axis.errorbar(
            x_positions + shift,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            linewidth=2,
            label=model,
        )

    y_max = _axis_y_max(
        profiles,
        class_index,
    )
    axis.set_ylim(
        0.0,
        y_max * 1.28,
    )

    axis.set_xticks(
        x_positions
    )
    axis.set_xticklabels([
        f"{name}\n{start:g}-{end:g} s"
        for name, start, end in TEMPORAL_WINDOWS
    ])
    axis.set_xlabel(
        "Temporal window"
    )
    axis.set_ylabel(
        "Mean normalized temporal relevance"
    )
    axis.grid(
        alpha=0.25,
    )


def _plot_values(
    profiles: list[SubjectProfile],
    model: str,
    class_index: int | None,
) -> np.ndarray:
    model_profiles = [
        profile
        for profile in profiles
        if profile.model == model
    ]

    if class_index is None:
        return np.stack([
            profile.values
            for profile in model_profiles
        ])

    return np.stack([
        profile.class_values[class_index]
        for profile in model_profiles
    ])


def _axis_y_max(
    profiles: list[SubjectProfile],
    class_index: int | None,
) -> float:
    upper_values = []

    for model in MODEL_ORDER:
        values = _plot_values(
            profiles,
            model,
            class_index,
        )
        upper_values.append(
            np.nanmean(
                values,
                axis=0,
            )
            + np.nanstd(
                values,
                axis=0,
                ddof=1,
            )
        )

    combined = np.concatenate(
        upper_values
    )

    if np.any(
        np.isfinite(
            combined
        )
    ):
        return max(
            float(
                np.nanmax(
                    combined
                )
            ),
            float(
                np.finfo(float).eps
            ),
        )

    return 1.0


def print_summary(
    title: str,
    summary_rows: list[SummaryRow],
    change_rows: list[ChangeRow],
    statistic_rows: list[StatisticRow],
) -> None:
    print()
    print(title)
    print("=" * len(title))
    print("Overall mean relevance across subjects")
    _print_summary_table(
        summary_rows,
        change_rows,
        "overall",
    )
    print()
    print("Class-wise mean relevance across subjects")
    _print_summary_table(
        summary_rows,
        change_rows,
        "classwise",
    )
    print()
    print("Paired Wilcoxon tests: weighted - baseline")
    print(
        f"{'Scope':<10} {'Class':<22} {'Window':<8} "
        f"{'Mean diff':>10} {'p':>10} {'q':>10} {'FDR':>5}"
    )
    for row in statistic_rows:
        print(
            f"{row.scope:<10} {row.class_name:<22} "
            f"{row.temporal_window:<8} "
            f"{row.mean_difference:>10.4f} "
            f"{row.p_value:>10.4g} "
            f"{row.p_value_fdr:>10.4g} "
            f"{str(row.significant_fdr):>5}"
        )


def _print_summary_table(
    summary_rows: list[SummaryRow],
    change_rows: list[ChangeRow],
    scope: str,
) -> None:
    print(
        f"{'Class':<22} {'Model':<24} "
        f"{'Early':>10} {'Middle':>10} {'Late':>10}"
    )
    for row in summary_rows:
        if row.scope != scope:
            continue
        print(
            f"{row.class_name:<22} {row.model:<24} "
            f"{row.early:>10.4f} {row.middle:>10.4f} "
            f"{row.late:>10.4f}"
        )

    for row in change_rows:
        if row.scope != scope:
            continue
        print(
            f"{row.class_name:<22} {'Weighted - Baseline':<24} "
            f"{row.early:>+10.4f} {row.middle:>+10.4f} "
            f"{row.late:>+10.4f}"
        )


def print_trial_counts(
    profiles: list[SubjectProfile],
) -> None:
    print()
    print("Trial counts by class")
    print(
        f"{'Selection':<15} {'Model':<24} {'Class':<12} "
        f"{'Mean':>8} {'Min':>5} {'Max':>5}"
    )

    for selection in (
        SELECTION_MAIN,
        SELECTION_SHARED,
    ):
        for model in MODEL_ORDER:
            model_profiles = [
                profile
                for profile in profiles
                if (
                    profile.selection == selection
                    and profile.model == model
                )
            ]

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                counts = np.asarray(
                    [
                        profile.class_counts[class_index]
                        for profile in model_profiles
                    ],
                    dtype=np.float64,
                )
                print(
                    f"{selection:<15} {model:<24} "
                    f"{class_name:<12} "
                    f"{np.mean(counts):>8.2f} "
                    f"{int(np.min(counts)):>5} "
                    f"{int(np.max(counts)):>5}"
                )


def _class_counts(
    labels: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(
        labels,
        dtype=int,
    )
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    return np.asarray(
        [
            np.sum(
                (labels == class_index)
                & mask
            )
            for class_index in range(len(CLASS_NAMES))
        ],
        dtype=int,
    )


def _summary_scopes() -> list[tuple[str, str]]:
    return [
        ("overall", "all_classes_balanced"),
        *[
            ("classwise", class_name)
            for class_name in CLASS_NAMES
        ],
    ]


def _find_summary(
    rows: list[SummaryRow],
    scope: str,
    class_name: str,
    model: str,
) -> SummaryRow:
    matches = [
        row
        for row in rows
        if (
            row.scope == scope
            and row.class_name == class_name
            and row.model == model
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one summary row for "
            f"{scope}, {class_name}, {model}."
        )

    return matches[0]


def _format_float(
    value: float,
) -> str:
    if not np.isfinite(
        value
    ):
        return ""

    return f"{value:.10g}"


if __name__ == "__main__":
    main()
