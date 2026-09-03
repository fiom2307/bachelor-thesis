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
    get_performance_accuracy_statistics_path,
    get_performance_class_recall_profiles_path,
    get_performance_class_recall_statistics_path,
    get_subject_name,
)
from src.utils.results import load_accuracy_comparison


ModelName = Literal[
    "csp_lda",
    "eegnet",
]


SUBJECTS = range(1, 10)
ALPHA = 0.05


@dataclass(frozen=True)
class AccuracyStatisticRow:
    n: int
    csp_mean: float
    csp_sd: float
    csp_median: float
    eegnet_mean: float
    eegnet_sd: float
    eegnet_median: float
    mean_difference: float
    wilcoxon_statistic: float
    p_value: float


@dataclass(frozen=True)
class SubjectClassRecallProfile:
    subject: int
    class_label: int
    class_name: str
    model: ModelName
    recall: float


@dataclass(frozen=True)
class ClassRecallStatisticRow:
    class_label: int
    class_name: str
    n: int
    csp_mean: float
    csp_sd: float
    eegnet_mean: float
    eegnet_sd: float
    mean_difference: float
    wilcoxon_statistic: float
    p_value: float
    p_value_fdr: float
    significant_fdr: bool
    direction: str


def run_performance_statistical_analysis() -> tuple[
    AccuracyStatisticRow,
    list[SubjectClassRecallProfile],
    list[ClassRecallStatisticRow],
]:
    """
    Run paired performance-level tests for CSP+LDA and EEGNet.
    """
    accuracy_row = compute_accuracy_statistics()
    recall_profiles = collect_subject_class_recall_profiles()
    recall_rows = compute_class_recall_statistics(
        recall_profiles
    )

    save_accuracy_statistics(
        accuracy_row
    )
    save_class_recall_profiles(
        recall_profiles
    )
    save_class_recall_statistics(
        recall_rows
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


def compute_accuracy_statistics() -> AccuracyStatisticRow:
    """
    Compute the planned paired Wilcoxon test for subject accuracies.
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

    differences = (
        eegnet_values
        - csp_values
    )

    statistic, p_value = _paired_wilcoxon(
        differences
    )

    return AccuracyStatisticRow(
        n=len(results),
        csp_mean=_mean(csp_values),
        csp_sd=_sample_sd(csp_values),
        csp_median=_median(csp_values),
        eegnet_mean=_mean(eegnet_values),
        eegnet_sd=_sample_sd(eegnet_values),
        eegnet_median=_median(eegnet_values),
        mean_difference=_mean(differences),
        wilcoxon_statistic=statistic,
        p_value=p_value,
    )


def collect_subject_class_recall_profiles() -> list[
    SubjectClassRecallProfile
]:
    """
    Compute one recall value per subject, class, and model.
    """
    profiles = []

    for subject in SUBJECTS:
        predictions = _load_subject_predictions(
            subject
        )

        y_true, csp_predictions, eegnet_predictions = predictions

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
) -> list[ClassRecallStatisticRow]:
    """
    Compute paired Wilcoxon tests per class and FDR-correct them.
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
        csp_values = csp_values[finite_mask]
        eegnet_values = eegnet_values[finite_mask]
        differences = (
            eegnet_values
            - csp_values
        )

        statistic, p_value = _paired_wilcoxon(
            differences
        )

        p_values.append(
            p_value
        )

        rows.append(
            ClassRecallStatisticRow(
                class_label=class_label,
                class_name=class_display_name,
                n=len(differences),
                csp_mean=_mean(csp_values),
                csp_sd=_sample_sd(csp_values),
                eegnet_mean=_mean(eegnet_values),
                eegnet_sd=_sample_sd(eegnet_values),
                mean_difference=_mean(differences),
                wilcoxon_statistic=statistic,
                p_value=p_value,
                p_value_fdr=np.nan,
                significant_fdr=False,
                direction=_direction_label(
                    _mean(differences)
                ),
            )
        )

    corrected_p_values, significant = _fdr_correct(
        p_values
    )

    return [
        ClassRecallStatisticRow(
            class_label=row.class_label,
            class_name=row.class_name,
            n=row.n,
            csp_mean=row.csp_mean,
            csp_sd=row.csp_sd,
            eegnet_mean=row.eegnet_mean,
            eegnet_sd=row.eegnet_sd,
            mean_difference=row.mean_difference,
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


def save_accuracy_statistics(
    row: AccuracyStatisticRow,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save the overall accuracy statistical test result.
    """
    if output_file is None:
        output_file = get_performance_accuracy_statistics_path()

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
                "n",
                "csp_mean",
                "csp_sd",
                "csp_median",
                "eegnet_mean",
                "eegnet_sd",
                "eegnet_median",
                "mean_difference",
                "wilcoxon_statistic",
                "p_value",
            ],
        )

        writer.writeheader()
        writer.writerow({
            "comparison": "CSP+LDA vs EEGNet accuracy",
            "n": row.n,
            "csp_mean": _format_float(row.csp_mean),
            "csp_sd": _format_float(row.csp_sd),
            "csp_median": _format_float(row.csp_median),
            "eegnet_mean": _format_float(row.eegnet_mean),
            "eegnet_sd": _format_float(row.eegnet_sd),
            "eegnet_median": _format_float(row.eegnet_median),
            "mean_difference": _format_float(
                row.mean_difference
            ),
            "wilcoxon_statistic": _format_float(
                row.wilcoxon_statistic
            ),
            "p_value": _format_float(row.p_value),
        })

    return output_file


def save_class_recall_profiles(
    profiles: list[SubjectClassRecallProfile],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save subject-level class-wise recall values.
    """
    if output_file is None:
        output_file = get_performance_class_recall_profiles_path()

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
            "class_label",
            "class_name",
            "model",
            "recall",
        ])

        for profile in profiles:
            writer.writerow([
                get_subject_name(profile.subject),
                profile.class_label,
                profile.class_name,
                profile.model,
                _format_float(profile.recall),
            ])

    return output_file


def save_class_recall_statistics(
    rows: list[ClassRecallStatisticRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save class-wise recall statistical test results.
    """
    if output_file is None:
        output_file = get_performance_class_recall_statistics_path()

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
                "class_label",
                "class_name",
                "n",
                "csp_mean",
                "csp_sd",
                "eegnet_mean",
                "eegnet_sd",
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
                "class_label": row.class_label,
                "class_name": row.class_name,
                "n": row.n,
                "csp_mean": _format_float(row.csp_mean),
                "csp_sd": _format_float(row.csp_sd),
                "eegnet_mean": _format_float(row.eegnet_mean),
                "eegnet_sd": _format_float(row.eegnet_sd),
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


def print_accuracy_statistics(
    row: AccuracyStatisticRow,
) -> None:
    """
    Print a concise summary of the overall accuracy comparison.
    """
    print()
    print("=" * 70)
    print("Overall accuracy comparison")
    print("=" * 70)
    print(
        "CSP+LDA: "
        f"mean={row.csp_mean:.4f}, "
        f"SD={row.csp_sd:.4f}, "
        f"median={row.csp_median:.4f}"
    )
    print(
        "EEGNet:  "
        f"mean={row.eegnet_mean:.4f}, "
        f"SD={row.eegnet_sd:.4f}, "
        f"median={row.eegnet_median:.4f}"
    )
    print(
        "EEGNet - CSP+LDA: "
        f"mean difference={row.mean_difference:.4f}, "
        f"W={row.wilcoxon_statistic:.4g}, "
        f"p={row.p_value:.4g}"
    )


def print_class_recall_statistics(
    rows: list[ClassRecallStatisticRow],
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
        f"{'CSP mean+/-SD':<18} "
        f"{'EEGNet mean+/-SD':<21} "
        f"{'Delta':>8} "
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
            f"{_mean_sd(row.csp_mean, row.csp_sd):<18} "
            f"{_mean_sd(row.eegnet_mean, row.eegnet_sd):<21} "
            f"{row.mean_difference:>8.4f} "
            f"{row.wilcoxon_statistic:>8.4g} "
            f"{row.p_value:>10.4g} "
            f"{row.p_value_fdr:>10.4g} "
            f"{str(row.significant_fdr):>5} "
            f"{row.direction}"
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
        differences
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
        return "EEGNet > CSP+LDA"

    if mean_difference < 0:
        return "EEGNet < CSP+LDA"

    return "EEGNet = CSP+LDA"


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
