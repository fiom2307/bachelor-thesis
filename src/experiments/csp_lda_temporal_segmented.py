from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS, CLASS_NAMES
from src.models.csp_lda import (
    load_csp_lda_fold_models,
    predict_csp_lda,
)
from src.models.lda import train_lda
from src.utils.config import (
    BASE_SEED,
    CSP_N_COMPONENTS,
    EPOCH_TMIN,
)
from src.utils.cross_validation import get_stratified_folds
from src.utils.paths import (
    get_csp_lda_temporal_segmented_class_summary_path,
    get_csp_lda_temporal_segmented_classification_report_path,
    get_csp_lda_temporal_segmented_confusion_matrix_path,
    get_csp_lda_temporal_segmented_subject_results_path,
    get_subject_name,
)


SFREQ = 250.0
ALPHA = 0.05
TEMPORAL_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("early", 0.5, 1.5),
    ("middle", 1.5, 2.5),
    ("late", 2.5, 4.0),
)
CLASS_KEYS = tuple(
    class_name.lower().replace(" ", "_")
    for class_name in CLASS_NAMES
)


@dataclass(frozen=True)
class SubjectTemporalSegmentedResult:
    subject: str
    baseline_accuracy: float
    temporal_accuracy: float
    accuracy_difference: float
    baseline_recalls: dict[str, float]
    temporal_recalls: dict[str, float]


@dataclass(frozen=True)
class TemporalSegmentedExperimentResult:
    subject_results: list[SubjectTemporalSegmentedResult]
    y_true: np.ndarray
    baseline_predictions: np.ndarray
    temporal_predictions: np.ndarray


@dataclass(frozen=True)
class TemporalSegmentedAccuracyStatistic:
    n_subjects: int
    baseline_mean: float
    baseline_sd: float
    temporal_mean: float
    temporal_sd: float
    mean_paired_difference: float
    wilcoxon_statistic: float
    p_value: float
    significant: bool


def run_temporal_segmented_accuracy_statistical_comparison() -> (
    TemporalSegmentedAccuracyStatistic
):
    """
    Compare saved baseline and temporal-segmented CSP+LDA accuracies.
    """
    rows = load_temporal_segmented_subject_results()
    statistic = compute_temporal_segmented_accuracy_statistic(
        rows
    )

    save_temporal_segmented_accuracy_statistic(
        statistic
    )
    print_temporal_segmented_accuracy_statistic(
        statistic
    )

    return statistic


def run_temporal_segmented_csp_lda_experiment() -> (
    TemporalSegmentedExperimentResult
):
    """
    Run the temporal-segmented CSP+LDA feature experiment.
    """
    subject_results = []
    all_y_true = []
    all_baseline_predictions = []
    all_temporal_predictions = []

    for subject in range(1, 10):
        subject_name = get_subject_name(subject)
        print(f"\nRunning {subject_name}...")

        data = get_data_for_subject(subject)

        if data is None:
            print(f"Skipping {subject_name}: data not found")
            continue

        (
            y_eval,
            baseline_predictions,
            temporal_predictions,
        ) = evaluate_temporal_segmented_csp_lda_for_subject(
            subject,
            data,
        )

        baseline_accuracy = float(
            accuracy_score(y_eval, baseline_predictions)
        )
        temporal_accuracy = float(
            accuracy_score(y_eval, temporal_predictions)
        )

        subject_results.append(
            SubjectTemporalSegmentedResult(
                subject=subject_name,
                baseline_accuracy=baseline_accuracy,
                temporal_accuracy=temporal_accuracy,
                accuracy_difference=temporal_accuracy - baseline_accuracy,
                baseline_recalls=_class_recalls(
                    y_eval,
                    baseline_predictions,
                ),
                temporal_recalls=_class_recalls(
                    y_eval,
                    temporal_predictions,
                ),
            )
        )

        all_y_true.append(y_eval)
        all_baseline_predictions.append(baseline_predictions)
        all_temporal_predictions.append(temporal_predictions)

    if not subject_results:
        raise RuntimeError(
            "No temporal-segmented CSP+LDA experiment results were generated."
        )

    result = TemporalSegmentedExperimentResult(
        subject_results=subject_results,
        y_true=np.concatenate(all_y_true),
        baseline_predictions=np.concatenate(all_baseline_predictions),
        temporal_predictions=np.concatenate(all_temporal_predictions),
    )

    save_temporal_segmented_subject_results(subject_results)
    save_temporal_segmented_class_summary(subject_results)
    save_temporal_segmented_pooled_report(result)
    save_temporal_segmented_confusion_matrices(result)
    print_temporal_segmented_summary(result)

    return result


def load_temporal_segmented_subject_results(
    input_file: str | Path | None = None,
) -> list[SubjectTemporalSegmentedResult]:
    """
    Load saved subject-wise temporal-segmented experiment results.
    """
    if input_file is None:
        input_file = (
            get_csp_lda_temporal_segmented_subject_results_path()
        )

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            "Saved temporal-segmented subject-wise results are missing: "
            f"{input_file}"
        )

    rows = []

    with input_file.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                SubjectTemporalSegmentedResult(
                    subject=row["subject"],
                    baseline_accuracy=float(
                        row["baseline_accuracy"]
                    ),
                    temporal_accuracy=float(
                        row["temporal_accuracy"]
                    ),
                    accuracy_difference=float(
                        row["accuracy_difference"]
                    ),
                    baseline_recalls={
                        "left_hand": float(
                            row["baseline_left_hand_recall"]
                        ),
                        "right_hand": float(
                            row["baseline_right_hand_recall"]
                        ),
                        "feet": float(
                            row["baseline_feet_recall"]
                        ),
                        "tongue": float(
                            row["baseline_tongue_recall"]
                        ),
                    },
                    temporal_recalls={
                        "left_hand": float(
                            row["temporal_left_hand_recall"]
                        ),
                        "right_hand": float(
                            row["temporal_right_hand_recall"]
                        ),
                        "feet": float(
                            row["temporal_feet_recall"]
                        ),
                        "tongue": float(
                            row["temporal_tongue_recall"]
                        ),
                    },
                )
            )

    return rows


def compute_temporal_segmented_accuracy_statistic(
    rows: list[SubjectTemporalSegmentedResult],
) -> TemporalSegmentedAccuracyStatistic:
    """
    Run a paired Wilcoxon test on saved subject-wise accuracies.
    """
    if not rows:
        raise ValueError(
            "No temporal-segmented subject-wise rows are available."
        )

    baseline = np.asarray(
        [
            row.baseline_accuracy
            for row in rows
        ],
        dtype=np.float64,
    )
    temporal = np.asarray(
        [
            row.temporal_accuracy
            for row in rows
        ],
        dtype=np.float64,
    )
    finite_mask = (
        np.isfinite(baseline)
        & np.isfinite(temporal)
    )
    baseline = baseline[finite_mask]
    temporal = temporal[finite_mask]

    if len(baseline) < 2:
        raise ValueError(
            "At least two paired subject accuracies are required."
        )

    statistic, p_value = wilcoxon(
        temporal,
        baseline,
        alternative="two-sided",
    )

    differences = temporal - baseline

    return TemporalSegmentedAccuracyStatistic(
        n_subjects=len(baseline),
        baseline_mean=float(
            np.mean(baseline)
        ),
        baseline_sd=float(
            np.std(
                baseline,
                ddof=1,
            )
        ),
        temporal_mean=float(
            np.mean(temporal)
        ),
        temporal_sd=float(
            np.std(
                temporal,
                ddof=1,
            )
        ),
        mean_paired_difference=float(
            np.mean(differences)
        ),
        wilcoxon_statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(
            p_value < ALPHA
        ),
    )


def save_temporal_segmented_accuracy_statistic(
    statistic: TemporalSegmentedAccuracyStatistic,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save the temporal-segmented vs baseline accuracy comparison.
    """
    if output_file is None:
        output_file = (
            get_csp_lda_temporal_segmented_subject_results_path()
            .with_name("accuracy_wilcoxon_statistics.csv")
        )

    output_file = Path(output_file)

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "comparison",
                "n_subjects",
                "baseline_mean",
                "baseline_sd",
                "temporal_mean",
                "temporal_sd",
                "mean_paired_difference",
                "wilcoxon_statistic",
                "p_value",
                "alpha",
                "significant",
            ],
        )

        writer.writeheader()
        writer.writerow({
            "comparison": (
                "baseline CSP+LDA vs temporal-segmented CSP+LDA"
            ),
            "n_subjects": statistic.n_subjects,
            "baseline_mean": _format_float(
                statistic.baseline_mean
            ),
            "baseline_sd": _format_float(
                statistic.baseline_sd
            ),
            "temporal_mean": _format_float(
                statistic.temporal_mean
            ),
            "temporal_sd": _format_float(
                statistic.temporal_sd
            ),
            "mean_paired_difference": _format_float(
                statistic.mean_paired_difference
            ),
            "wilcoxon_statistic": _format_float(
                statistic.wilcoxon_statistic
            ),
            "p_value": _format_float(
                statistic.p_value
            ),
            "alpha": _format_float(ALPHA),
            "significant": statistic.significant,
        })

    return output_file


def evaluate_temporal_segmented_csp_lda_for_subject(
    subject: int,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return labels, baseline predictions, and temporal-feature predictions.
    """
    x_train, y_train, x_eval, y_eval = data
    fold_probabilities = []
    baseline_models = []

    for fold, train_idx, _ in get_stratified_folds(
        x_train,
        y_train,
        BASE_SEED + subject,
    ):
        saved_models = load_csp_lda_fold_models(subject, fold)

        if saved_models is None:
            raise FileNotFoundError(
                "Missing saved baseline CSP+LDA fold models for "
                f"{get_subject_name(subject)} fold {fold}."
            )

        csp, lda = saved_models
        baseline_models.append((csp, lda))

        x_train_features = extract_temporal_segmented_csp_features(
            csp,
            x_train[train_idx],
        )
        temporal_lda = train_lda(
            x_train_features,
            y_train[train_idx],
        )
        x_eval_features = extract_temporal_segmented_csp_features(
            csp,
            x_eval,
        )

        fold_probabilities.append(
            _probabilities_for_all_classes(
                temporal_lda,
                x_eval_features,
            )
        )

    baseline_predictions = predict_csp_lda(
        baseline_models,
        x_eval,
    )
    temporal_predictions = np.asarray(CLASS_LABELS)[
        np.argmax(
            np.mean(fold_probabilities, axis=0),
            axis=1,
        )
    ]

    return (
        np.asarray(y_eval),
        np.asarray(baseline_predictions),
        np.asarray(temporal_predictions),
    )


def extract_temporal_segmented_csp_features(
    csp,
    x: np.ndarray,
) -> np.ndarray:
    """
    Apply saved CSP filters and compute log-variance per temporal window.
    """
    filters = np.asarray(
        csp.filters_[:CSP_N_COMPONENTS],
        dtype=np.float64,
    )
    component_time_series = np.einsum(
        "kc,nct->nkt",
        filters,
        x,
    )
    times = _create_times(x.shape[-1])
    feature_blocks = []

    for _, start, end in TEMPORAL_WINDOWS:
        mask = _time_window_mask(times, start, end)

        if not np.any(mask):
            raise ValueError(
                f"No samples found for temporal window {start}-{end} s."
            )

        window_values = component_time_series[:, :, mask]
        feature_blocks.append(
            np.log(
                np.mean(
                    window_values ** 2,
                    axis=2,
                )
            )
        )

    return np.concatenate(feature_blocks, axis=1)


def save_temporal_segmented_subject_results(
    rows: list[SubjectTemporalSegmentedResult],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save subject-wise temporal-segmented experiment results.
    """
    if output_file is None:
        output_file = (
            get_csp_lda_temporal_segmented_subject_results_path()
        )

    output_file = Path(output_file)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "baseline_accuracy",
                "temporal_accuracy",
                "accuracy_difference",
                "baseline_left_hand_recall",
                "temporal_left_hand_recall",
                "baseline_right_hand_recall",
                "temporal_right_hand_recall",
                "baseline_feet_recall",
                "temporal_feet_recall",
                "baseline_tongue_recall",
                "temporal_tongue_recall",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "subject": row.subject,
                "baseline_accuracy": _format_float(
                    row.baseline_accuracy
                ),
                "temporal_accuracy": _format_float(
                    row.temporal_accuracy
                ),
                "accuracy_difference": _format_float(
                    row.accuracy_difference
                ),
                "baseline_left_hand_recall": _format_float(
                    row.baseline_recalls["left_hand"]
                ),
                "temporal_left_hand_recall": _format_float(
                    row.temporal_recalls["left_hand"]
                ),
                "baseline_right_hand_recall": _format_float(
                    row.baseline_recalls["right_hand"]
                ),
                "temporal_right_hand_recall": _format_float(
                    row.temporal_recalls["right_hand"]
                ),
                "baseline_feet_recall": _format_float(
                    row.baseline_recalls["feet"]
                ),
                "temporal_feet_recall": _format_float(
                    row.temporal_recalls["feet"]
                ),
                "baseline_tongue_recall": _format_float(
                    row.baseline_recalls["tongue"]
                ),
                "temporal_tongue_recall": _format_float(
                    row.temporal_recalls["tongue"]
                ),
            })

    return output_file


def save_temporal_segmented_class_summary(
    rows: list[SubjectTemporalSegmentedResult],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save class-wise recall summaries across subjects.
    """
    if output_file is None:
        output_file = get_csp_lda_temporal_segmented_class_summary_path()

    output_file = Path(output_file)

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_name",
                "baseline_mean_recall",
                "baseline_sd_recall",
                "temporal_mean_recall",
                "temporal_sd_recall",
                "mean_difference",
            ],
        )
        writer.writeheader()

        for class_name in CLASS_NAMES:
            class_key = _class_key(
                class_name
            )
            baseline_values = np.asarray(
                [row.baseline_recalls[class_key] for row in rows],
                dtype=np.float64,
            )
            temporal_values = np.asarray(
                [row.temporal_recalls[class_key] for row in rows],
                dtype=np.float64,
            )

            writer.writerow({
                "class_name": class_name.title(),
                "baseline_mean_recall": _format_float(
                    np.mean(baseline_values)
                ),
                "baseline_sd_recall": _format_float(
                    np.std(baseline_values, ddof=1)
                ),
                "temporal_mean_recall": _format_float(
                    np.mean(temporal_values)
                ),
                "temporal_sd_recall": _format_float(
                    np.std(temporal_values, ddof=1)
                ),
                "mean_difference": _format_float(
                    np.mean(temporal_values - baseline_values)
                ),
            })

    return output_file


def save_temporal_segmented_pooled_report(
    result: TemporalSegmentedExperimentResult,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save pooled classification reports for baseline and temporal CSP+LDA.
    """
    if output_file is None:
        output_file = (
            get_csp_lda_temporal_segmented_classification_report_path()
        )

    output_file = Path(output_file)

    baseline_report = classification_report(
        result.y_true,
        result.baseline_predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    temporal_report = classification_report(
        result.y_true,
        result.temporal_predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    with output_file.open("w", encoding="utf-8") as file:
        file.write("Baseline CSP+LDA - All subjects\n")
        file.write("=" * 70)
        file.write("\n")
        file.write(baseline_report)
        file.write("\n\nTemporal segmented CSP+LDA - All subjects\n")
        file.write("=" * 70)
        file.write("\n")
        file.write(temporal_report)

    return output_file


def save_temporal_segmented_confusion_matrices(
    result: TemporalSegmentedExperimentResult,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save pooled confusion matrices for baseline and temporal CSP+LDA.
    """
    if output_file is None:
        output_file = get_csp_lda_temporal_segmented_confusion_matrix_path()

    output_file = Path(output_file)
    baseline_matrix = confusion_matrix(
        result.y_true,
        result.baseline_predictions,
        labels=CLASS_LABELS,
    )
    temporal_matrix = confusion_matrix(
        result.y_true,
        result.temporal_predictions,
        labels=CLASS_LABELS,
    )

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "true_class",
            *[f"predicted_{class_name}" for class_name in CLASS_NAMES],
        ])

        for model_name, matrix in (
            ("baseline_csp_lda", baseline_matrix),
            ("temporal_segmented_csp_lda", temporal_matrix),
        ):
            for class_name, values in zip(
                CLASS_NAMES,
                matrix,
                strict=True,
            ):
                writer.writerow([
                    model_name,
                    class_name,
                    *values.tolist(),
                ])

    return output_file


def print_temporal_segmented_summary(
    result: TemporalSegmentedExperimentResult,
) -> None:
    """
    Print subject-wise, aggregate, class-wise, and pooled summaries.
    """
    rows = result.subject_results

    print()
    print("=" * 70)
    print("CSP+LDA temporal-segmented feature experiment")
    print("=" * 70)
    print(
        f"{'Subject':<10} "
        f"{'Baseline':>10} "
        f"{'Temporal':>10} "
        f"{'Delta':>10}"
    )

    for row in rows:
        print(
            f"{row.subject:<10} "
            f"{row.baseline_accuracy:>10.4f} "
            f"{row.temporal_accuracy:>10.4f} "
            f"{row.accuracy_difference:>10.4f}"
        )

    baseline_accuracies = np.asarray(
        [row.baseline_accuracy for row in rows],
        dtype=np.float64,
    )
    temporal_accuracies = np.asarray(
        [row.temporal_accuracy for row in rows],
        dtype=np.float64,
    )
    differences = temporal_accuracies - baseline_accuracies

    print("-" * 70)
    print(
        "Baseline mean+/-SD: "
        f"{np.mean(baseline_accuracies):.4f} +/- "
        f"{np.std(baseline_accuracies, ddof=1):.4f}"
    )
    print(
        "Temporal mean+/-SD: "
        f"{np.mean(temporal_accuracies):.4f} +/- "
        f"{np.std(temporal_accuracies, ddof=1):.4f}"
    )
    print(f"Mean difference: {np.mean(differences):.4f}")

    print()
    print("Class-wise recall across subjects")
    print(
        f"{'Class':<12} "
        f"{'Baseline mean+/-SD':<22} "
        f"{'Temporal mean+/-SD':<22} "
        f"{'Delta':>10}"
    )

    for class_name in CLASS_NAMES:
        class_key = _class_key(
            class_name
        )
        baseline_values = np.asarray(
            [row.baseline_recalls[class_key] for row in rows],
            dtype=np.float64,
        )
        temporal_values = np.asarray(
            [row.temporal_recalls[class_key] for row in rows],
            dtype=np.float64,
        )
        print(
            f"{class_name.title():<12} "
            f"{_mean_sd(baseline_values):<22} "
            f"{_mean_sd(temporal_values):<22} "
            f"{np.mean(temporal_values - baseline_values):>10.4f}"
        )

    print()
    print("Baseline CSP+LDA - All subjects")
    print(
        classification_report(
            result.y_true,
            result.baseline_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("Temporal segmented CSP+LDA - All subjects")
    print(
        classification_report(
            result.y_true,
            result.temporal_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("Baseline confusion matrix")
    print(
        confusion_matrix(
            result.y_true,
            result.baseline_predictions,
            labels=CLASS_LABELS,
        )
    )
    print("Temporal segmented confusion matrix")
    print(
        confusion_matrix(
            result.y_true,
            result.temporal_predictions,
            labels=CLASS_LABELS,
        )
    )


def print_temporal_segmented_accuracy_statistic(
    statistic: TemporalSegmentedAccuracyStatistic,
) -> None:
    """
    Print the paired accuracy comparison summary.
    """
    print()
    print("=" * 70)
    print("Baseline vs temporal-segmented CSP+LDA accuracy")
    print("=" * 70)
    print(
        f"n={statistic.n_subjects}, "
        "baseline="
        f"{statistic.baseline_mean:.4f} +/- "
        f"{statistic.baseline_sd:.4f}, "
        "temporal="
        f"{statistic.temporal_mean:.4f} +/- "
        f"{statistic.temporal_sd:.4f}"
    )
    print(
        "temporal - baseline="
        f"{statistic.mean_paired_difference:.4f}, "
        f"W={statistic.wilcoxon_statistic:.4g}, "
        f"p={statistic.p_value:.4g}, "
        f"significant={statistic.significant}"
    )


def _class_recalls(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    recalls = {}

    for class_label, class_key in zip(
        CLASS_LABELS,
        CLASS_KEYS,
        strict=True,
    ):
        class_mask = y_true == class_label
        recalls[class_key] = float(
            np.mean(y_pred[class_mask] == class_label)
        )

    return recalls


def _class_key(
    class_name: str,
) -> str:
    return class_name.lower().replace(
        " ",
        "_",
    )


def _probabilities_for_all_classes(
    lda,
    features: np.ndarray,
) -> np.ndarray:
    probabilities = lda.predict_proba(features)
    aligned_probabilities = np.zeros(
        (
            len(features),
            len(CLASS_LABELS),
        ),
        dtype=np.float64,
    )

    for class_index, class_label in enumerate(lda.classes_):
        output_index = CLASS_LABELS.index(int(class_label))
        aligned_probabilities[:, output_index] = probabilities[
            :,
            class_index,
        ]

    return aligned_probabilities


def _create_times(
    n_times: int,
) -> np.ndarray:
    return EPOCH_TMIN + np.arange(n_times) / SFREQ


def _time_window_mask(
    times: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    if end == TEMPORAL_WINDOWS[-1][2]:
        return (times >= start) & (times <= end)

    return (times >= start) & (times < end)


def _mean_sd(
    values: np.ndarray,
) -> str:
    return (
        f"{np.mean(values):.4f} +/- "
        f"{np.std(values, ddof=1):.4f}"
    )


def _format_float(
    value: float,
) -> str:
    return f"{value:.10g}"
