from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS, CLASS_NAMES
from src.data.preprocessing import (
    normalize_epochs,
    prepare_eegnet_input,
)
from src.models.csp_lda import (
    load_csp_lda_fold_models,
    predict_csp_lda,
)
from src.models.eegnet import predict_eegnet
from src.utils.config import N_FOLDS
from src.utils.paths import (
    get_eegnet_fold_model_path,
    get_performance_classwise_results_path,
    get_performance_overall_results_path,
    get_subject_name,
)
from src.utils.results import load_accuracy_comparison


ModelName = Literal[
    "csp_lda",
    "eegnet",
]


SUBJECTS = range(1, 10)
ALPHA = 0.05
COMPARISON_NAME = "CSP+LDA vs EEGNet"
COMPARISON_SLUG = "csp_lda_vs_eegnet"
LEFT_LABEL = "CSP+LDA"
RIGHT_LABEL = "EEGNet"


@dataclass(frozen=True)
class PerformanceStatisticRow:
    comparison: str
    comparison_slug: str
    scope: str
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
    direction: str
    class_name: str | None = None
    p_value_fdr: float = np.nan
    significant_fdr: bool = False


@dataclass(frozen=True)
class SubjectClassRecallProfile:
    subject: int
    class_label: int
    class_name: str
    model: ModelName
    recall: float


def run_performance_statistical_analysis() -> tuple[
    PerformanceStatisticRow,
    list[SubjectClassRecallProfile],
    list[PerformanceStatisticRow],
]:
    """
    Run paired performance-level tests for CSP+LDA and EEGNet.
    """
    accuracy_row = compute_accuracy_statistics()
    recall_profiles = collect_subject_class_recall_profiles()
    recall_rows = compute_class_recall_statistics(
        recall_profiles
    )

    save_performance_statistics(
        accuracy_row=accuracy_row,
        recall_rows=recall_rows,
    )

    print_accuracy_statistics(
        accuracy_row
    )
    print_class_recall_statistics(
        recall_rows
    )

    return (
        accuracy_row,
        recall_profiles,
        recall_rows,
    )


def compute_accuracy_statistics() -> PerformanceStatisticRow:
    """
    Compute the existing paired Wilcoxon test for subject accuracies.
    """
    results = load_accuracy_comparison()

    if results is None:
        raise FileNotFoundError(
            "Accuracy comparison CSV not found. Run "
            "`python -m scripts.compare_model_accuracies` first."
        )

    if len(results) != len(list(SUBJECTS)):
        raise ValueError(
            "Expected accuracy results for 9 subjects, "
            f"but found {len(results)}."
        )

    expected_subjects = [
        get_subject_name(subject)
        for subject in SUBJECTS
    ]
    actual_subjects = [
        row[0]
        for row in results
    ]

    if actual_subjects != expected_subjects:
        raise ValueError(
            "Expected accuracy rows for A01-A09 in order, "
            f"but found {actual_subjects}."
        )

    csp_values = np.asarray(
        [
            row[1]
            for row in results
        ],
        dtype=np.float64,
    )
    eegnet_values = np.asarray(
        [
            row[2]
            for row in results
        ],
        dtype=np.float64,
    )

    return _compute_statistic_row(
        scope="overall",
        metric="accuracy",
        class_name=None,
        left_values=csp_values,
        right_values=eegnet_values,
    )


def collect_subject_class_recall_profiles() -> list[
    SubjectClassRecallProfile
]:
    """
    Compute one recall value per subject, class, and model.
    """
    profiles = []

    for subject in SUBJECTS:
        y_true, csp_predictions, eegnet_predictions = (
            _load_subject_predictions(
                subject
            )
        )

        for class_label, class_name in zip(
            CLASS_LABELS,
            CLASS_NAMES,
            strict=True,
        ):
            class_display_name = _display_class_name(
                class_name
            )

            profiles.append(
                SubjectClassRecallProfile(
                    subject=subject,
                    class_label=class_label,
                    class_name=class_display_name,
                    model="csp_lda",
                    recall=_class_recall(
                        y_true,
                        csp_predictions,
                        class_label,
                    ),
                )
            )

            profiles.append(
                SubjectClassRecallProfile(
                    subject=subject,
                    class_label=class_label,
                    class_name=class_display_name,
                    model="eegnet",
                    recall=_class_recall(
                        y_true,
                        eegnet_predictions,
                        class_label,
                    ),
                )
            )

    return profiles


def compute_class_recall_statistics(
    profiles: list[SubjectClassRecallProfile],
) -> list[PerformanceStatisticRow]:
    """
    Compute paired Wilcoxon tests per class and FDR-correct the 4 tests.
    """
    profile_lookup = {
        (
            profile.subject,
            profile.class_label,
            profile.model,
        ): profile.recall
        for profile in profiles
    }

    rows = []
    p_values = []

    for class_label, class_name in zip(
        CLASS_LABELS,
        CLASS_NAMES,
        strict=True,
    ):
        class_display_name = _display_class_name(
            class_name
        )

        csp_values = np.asarray(
            [
                profile_lookup[
                    (subject, class_label, "csp_lda")
                ]
                for subject in SUBJECTS
            ],
            dtype=np.float64,
        )
        eegnet_values = np.asarray(
            [
                profile_lookup[
                    (subject, class_label, "eegnet")
                ]
                for subject in SUBJECTS
            ],
            dtype=np.float64,
        )

        finite_mask = (
            np.isfinite(csp_values)
            & np.isfinite(eegnet_values)
        )

        row = _compute_statistic_row(
            scope="classwise",
            metric="recall",
            class_name=class_display_name,
            left_values=csp_values[finite_mask],
            right_values=eegnet_values[finite_mask],
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
        PerformanceStatisticRow(
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


def save_performance_statistics(
    accuracy_row: PerformanceStatisticRow,
    recall_rows: list[PerformanceStatisticRow],
) -> list[Path]:
    """
    Save overall_statistics.csv and classwise_statistics.csv.
    """
    overall_path = save_accuracy_statistics(
        accuracy_row
    )
    classwise_path = save_class_recall_statistics(
        recall_rows
    )

    return [
        overall_path,
        classwise_path,
    ]


def save_accuracy_statistics(
    row: PerformanceStatisticRow,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save the overall accuracy statistical test result.
    """
    if output_file is None:
        output_file = get_performance_overall_results_path()

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
                "direction",
            ],
        )

        writer.writeheader()
        writer.writerow({
            "comparison": row.comparison,
            "scope": row.scope,
            "metric": row.metric,
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
            "direction": row.direction,
        })

    return output_file


def save_class_recall_statistics(
    rows: list[PerformanceStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save class-wise recall statistical test results.
    """
    if output_file is None:
        output_file = get_performance_classwise_results_path()

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


def print_accuracy_statistics(
    row: PerformanceStatisticRow,
) -> None:
    """
    Print a concise summary of the overall accuracy comparison.
    """
    print()
    print("=" * 70)
    print("Overall accuracy comparison")
    print("=" * 70)
    print(
        f"{LEFT_LABEL}: "
        f"mean={row.mean_a:.4f}, "
        f"SD={row.std_a:.4f}, "
        f"median={row.median_a:.4f}"
    )
    print(
        f"{RIGHT_LABEL}:  "
        f"mean={row.mean_b:.4f}, "
        f"SD={row.std_b:.4f}, "
        f"median={row.median_b:.4f}"
    )
    print(
        f"{LEFT_LABEL} - {RIGHT_LABEL}: "
        f"mean difference={row.mean_difference:.4f}, "
        f"W={row.wilcoxon_statistic:.4g}, "
        f"p={row.p_value:.4g}"
    )


def print_class_recall_statistics(
    rows: list[PerformanceStatisticRow],
) -> None:
    """
    Print a concise class-wise recall comparison.
    """
    print()
    print("=" * 70)
    print("Class-wise recall comparison")
    print("=" * 70)
    print(
        f"{'Class':<12} "
        f"{'n':>2} "
        f"{LEFT_LABEL + ' mean+/-SD':<20} "
        f"{RIGHT_LABEL + ' mean+/-SD':<21} "
        f"{'A-B':>8} "
        f"{'W':>8} "
        f"{'p':>10} "
        f"{'p_FDR':>10} "
        f"{'Sig':>5} "
        "Direction"
    )

    for row in rows:
        print(
            f"{row.class_name:<12} "
            f"{row.n:>2} "
            f"{_mean_sd(row.mean_a, row.std_a):<20} "
            f"{_mean_sd(row.mean_b, row.std_b):<21} "
            f"{row.mean_difference:>8.4f} "
            f"{row.wilcoxon_statistic:>8.4g} "
            f"{row.p_value:>10.4g} "
            f"{row.p_value_fdr:>10.4g} "
            f"{str(row.significant_fdr):>5} "
            f"{row.direction}"
        )


def _compute_statistic_row(
    scope: str,
    metric: str,
    class_name: str | None,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> PerformanceStatisticRow:
    differences = (
        left_values
        - right_values
    )

    statistic, p_value = _paired_wilcoxon(
        differences
    )

    mean_difference = _mean(
        differences
    )

    return PerformanceStatisticRow(
        comparison=COMPARISON_NAME,
        comparison_slug=COMPARISON_SLUG,
        scope=scope,
        class_name=class_name,
        metric=metric,
        n=len(differences),
        mean_a=_mean(left_values),
        std_a=_sample_sd(left_values),
        median_a=_median(left_values),
        mean_b=_mean(right_values),
        std_b=_sample_sd(right_values),
        median_b=_median(right_values),
        mean_difference=mean_difference,
        median_difference=_median(
            differences
        ),
        wilcoxon_statistic=statistic,
        p_value=p_value,
        direction=_direction_label(
            mean_difference
        ),
    )


def _load_subject_predictions(
    subject: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load saved ensembles and return final predictions for one subject.
    """
    data = get_data_for_subject(
        subject
    )

    if data is None:
        raise FileNotFoundError(
            f"Data for {get_subject_name(subject)} not found."
        )

    X_train, _, X_eval, y_eval = data

    csp_models = _load_saved_csp_lda_ensemble(
        subject
    )
    eegnet_models = _load_saved_eegnet_ensemble(
        subject
    )

    _, X_eval_eegnet = normalize_epochs(
        X_train,
        X_eval,
    )
    X_eval_eegnet = prepare_eegnet_input(
        X_eval_eegnet
    )

    csp_predictions = predict_csp_lda(
        csp_models,
        X_eval,
    )
    eegnet_predictions = predict_eegnet(
        eegnet_models,
        X_eval_eegnet,
    )

    return (
        np.asarray(y_eval),
        np.asarray(csp_predictions),
        np.asarray(eegnet_predictions),
    )


def _load_saved_csp_lda_ensemble(
    subject: int,
):
    models = []
    missing = []

    for fold in range(1, N_FOLDS + 1):
        fold_models = load_csp_lda_fold_models(
            subject,
            fold,
        )

        if fold_models is None:
            missing.append(
                f"fold {fold}"
            )
            continue

        models.append(
            fold_models
        )

    if missing:
        raise FileNotFoundError(
            f"Missing saved CSP+LDA models for "
            f"{get_subject_name(subject)}: {', '.join(missing)}."
        )

    return models


def _load_saved_eegnet_ensemble(
    subject: int,
) -> list[tf.keras.Model]:
    models = []
    missing = []

    for fold in range(1, N_FOLDS + 1):
        model_path = get_eegnet_fold_model_path(
            subject,
            fold,
        )

        if not model_path.exists():
            missing.append(
                str(model_path)
            )
            continue

        models.append(
            tf.keras.models.load_model(
                model_path
            )
        )

    if missing:
        raise FileNotFoundError(
            "Missing saved EEGNet models for "
            f"{get_subject_name(subject)}: "
            f"{'; '.join(missing)}."
        )

    return models


def _class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_label: int,
) -> float:
    class_mask = (
        y_true
        == class_label
    )

    n_class_trials = int(
        np.sum(
            class_mask
        )
    )

    if n_class_trials == 0:
        return np.nan

    true_positives = int(
        np.sum(
            y_pred[class_mask]
            == class_label
        )
    )

    return (
        true_positives
        / n_class_trials
    )


def _paired_wilcoxon(
    differences: np.ndarray,
) -> tuple[float, float]:
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


def _fdr_correct(
    p_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
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
            p_values_array[finite_mask],
            alpha=ALPHA,
            method="fdr_bh",
        )

        corrected[finite_mask] = corrected_values
        significant[finite_mask] = reject

    return corrected, significant


def _direction_label(
    mean_difference: float,
) -> str:
    if not np.isfinite(
        mean_difference
    ):
        return "not available"

    if mean_difference > 0:
        return f"{LEFT_LABEL} > {RIGHT_LABEL}"

    if mean_difference < 0:
        return f"{LEFT_LABEL} < {RIGHT_LABEL}"

    return f"{LEFT_LABEL} = {RIGHT_LABEL}"


def _display_class_name(
    class_name: str,
) -> str:
    return class_name.title()


def _mean(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return np.nan

    return float(
        np.nanmean(
            values
        )
    )


def _median(
    values: np.ndarray,
) -> float:
    if len(values) == 0:
        return np.nan

    return float(
        np.nanmedian(
            values
        )
    )


def _sample_sd(
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
    sd: float,
) -> str:
    return f"{mean:.4f}+/-{sd:.4f}"


def _format_float(
    value: float,
) -> str:
    if not np.isfinite(
        value
    ):
        return ""

    return f"{value:.10g}"
