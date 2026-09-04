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
    get_csp_lda_early_weighted_accuracy_statistics_path,
    get_csp_lda_early_weighted_class_summary_path,
    get_csp_lda_early_weighted_classification_report_path,
    get_csp_lda_early_weighted_confusion_matrix_path,
    get_csp_lda_early_weighted_selected_lambdas_path,
    get_csp_lda_early_weighted_subject_results_path,
    get_subject_name,
)


SFREQ = 250.0
ALPHA = 0.05
FEATURE_EPS = np.finfo(np.float64).eps

TEMPORAL_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("early", 0.5, 1.5),
    ("middle", 1.5, 2.5),
    ("late", 2.5, 4.0),
)
LAMBDA_EARLY_VALUES = (
    1.0,
    1.25,
    1.5,
    2.0,
)
CLASS_KEYS = tuple(
    class_name.lower().replace(" ", "_")
    for class_name in CLASS_NAMES
)


@dataclass(frozen=True)
class SelectedLambdaRow:
    subject: str
    fold: int
    lambda_early: float
    validation_accuracy: float


@dataclass(frozen=True)
class SubjectEarlyWeightedResult:
    subject: str
    baseline_accuracy: float
    early_weighted_accuracy: float
    accuracy_difference: float
    baseline_recalls: dict[str, float]
    early_weighted_recalls: dict[str, float]


@dataclass(frozen=True)
class EarlyWeightedExperimentResult:
    subject_results: list[SubjectEarlyWeightedResult]
    selected_lambdas: list[SelectedLambdaRow]
    y_true: np.ndarray
    baseline_predictions: np.ndarray
    early_weighted_predictions: np.ndarray


@dataclass(frozen=True)
class EarlyWeightedAccuracyStatistic:
    n_subjects: int
    baseline_mean: float
    baseline_sd: float
    early_weighted_mean: float
    early_weighted_sd: float
    mean_paired_difference: float
    wilcoxon_statistic: float
    p_value: float
    significant: bool


def run_csp_lda_early_weighted_experiment() -> EarlyWeightedExperimentResult:
    """
    Run the Early-weighted CSP+LDA experiment.
    """
    subject_results = []
    selected_lambdas = []
    all_y_true = []
    all_baseline_predictions = []
    all_early_weighted_predictions = []

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )
        print(
            f"\nRunning {subject_name}..."
        )

        data = get_data_for_subject(
            subject
        )

        if data is None:
            print(
                f"Skipping {subject_name}: data not found"
            )
            continue

        (
            y_eval,
            baseline_predictions,
            early_weighted_predictions,
            subject_lambdas,
        ) = evaluate_csp_lda_early_weighted_for_subject(
            subject,
            data,
        )

        baseline_accuracy = float(
            accuracy_score(
                y_eval,
                baseline_predictions,
            )
        )
        early_weighted_accuracy = float(
            accuracy_score(
                y_eval,
                early_weighted_predictions,
            )
        )

        subject_results.append(
            SubjectEarlyWeightedResult(
                subject=subject_name,
                baseline_accuracy=baseline_accuracy,
                early_weighted_accuracy=early_weighted_accuracy,
                accuracy_difference=(
                    early_weighted_accuracy
                    - baseline_accuracy
                ),
                baseline_recalls=_class_recalls(
                    y_eval,
                    baseline_predictions,
                ),
                early_weighted_recalls=_class_recalls(
                    y_eval,
                    early_weighted_predictions,
                ),
            )
        )
        selected_lambdas.extend(
            subject_lambdas
        )
        all_y_true.append(
            y_eval
        )
        all_baseline_predictions.append(
            baseline_predictions
        )
        all_early_weighted_predictions.append(
            early_weighted_predictions
        )

    if not subject_results:
        raise RuntimeError(
            "No Early-weighted CSP+LDA experiment results were generated."
        )

    result = EarlyWeightedExperimentResult(
        subject_results=subject_results,
        selected_lambdas=selected_lambdas,
        y_true=np.concatenate(
            all_y_true
        ),
        baseline_predictions=np.concatenate(
            all_baseline_predictions
        ),
        early_weighted_predictions=np.concatenate(
            all_early_weighted_predictions
        ),
    )
    statistic = compute_accuracy_statistic(
        subject_results
    )

    save_subject_results(
        subject_results
    )
    save_selected_lambdas(
        selected_lambdas
    )
    save_class_summary(
        subject_results
    )
    save_pooled_classification_report(
        result
    )
    save_pooled_confusion_matrix(
        result
    )
    save_accuracy_statistic(
        statistic
    )

    print_experiment_summary(
        result,
        statistic,
    )

    return result


def evaluate_csp_lda_early_weighted_for_subject(
    subject: int,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[SelectedLambdaRow],
]:
    """
    Evaluate baseline and Early-weighted CSP+LDA for one subject.
    """
    x_train, y_train, x_eval, y_eval = data
    baseline_models = []
    fold_probabilities = []
    selected_lambdas = []
    subject_name = get_subject_name(
        subject
    )

    for fold, train_idx, validation_idx in get_stratified_folds(
        x_train,
        y_train,
        BASE_SEED + subject,
    ):
        saved_models = load_csp_lda_fold_models(
            subject,
            fold,
        )

        if saved_models is None:
            raise FileNotFoundError(
                "Missing saved baseline CSP+LDA fold models for "
                f"{subject_name} fold {fold}."
            )

        csp, baseline_lda = saved_models
        baseline_models.append(
            (
                csp,
                baseline_lda,
            )
        )

        x_train_features = extract_temporal_csp_features(
            csp,
            x_train[train_idx],
        )
        lda = train_lda(
            x_train_features,
            y_train[train_idx],
        )

        x_validation_features = extract_temporal_csp_features(
            csp,
            x_train[validation_idx],
        )
        lambda_early, validation_accuracy = select_lambda_early(
            lda=lda,
            features=x_validation_features,
            labels=y_train[validation_idx],
        )

        selected_lambdas.append(
            SelectedLambdaRow(
                subject=subject_name,
                fold=fold,
                lambda_early=lambda_early,
                validation_accuracy=validation_accuracy,
            )
        )

        x_eval_features = extract_temporal_csp_features(
            csp,
            x_eval,
        )
        fold_probabilities.append(
            predict_proba_early_weighted_lda(
                lda=lda,
                features=x_eval_features,
                lambda_early=lambda_early,
            )
        )

    baseline_predictions = predict_csp_lda(
        baseline_models,
        x_eval,
    )
    early_weighted_predictions = np.asarray(
        CLASS_LABELS
    )[
        np.argmax(
            np.mean(
                fold_probabilities,
                axis=0,
            ),
            axis=1,
        )
    ]

    return (
        np.asarray(
            y_eval
        ),
        np.asarray(
            baseline_predictions
        ),
        np.asarray(
            early_weighted_predictions
        ),
        selected_lambdas,
    )


def extract_temporal_csp_features(
    csp,
    x: np.ndarray,
) -> np.ndarray:
    """
    Apply saved CSP filters and compute 12 windowed log-power features.
    """
    filters = np.asarray(
        csp.filters_[:CSP_N_COMPONENTS],
        dtype=np.float64,
    )
    component_time_series = np.einsum(
        "kc,nct->nkt",
        filters,
        np.asarray(
            x,
            dtype=np.float64,
        ),
    )
    times = _create_times(
        x.shape[-1]
    )
    feature_blocks = []

    for _, start, end in TEMPORAL_WINDOWS:
        mask = _time_window_mask(
            times,
            start,
            end,
        )

        if not np.any(
            mask
        ):
            raise ValueError(
                f"No samples found for temporal window {start}-{end} s."
            )

        window_values = component_time_series[
            :,
            :,
            mask,
        ]
        feature_blocks.append(
            np.log(
                np.maximum(
                    np.mean(
                        window_values**2,
                        axis=2,
                    ),
                    FEATURE_EPS,
                )
            )
        )

    return np.concatenate(
        feature_blocks,
        axis=1,
    )


def select_lambda_early(
    lda,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """
    Select lambda_early using only fold validation data.
    """
    best_lambda = LAMBDA_EARLY_VALUES[0]
    best_accuracy = -np.inf

    for lambda_early in LAMBDA_EARLY_VALUES:
        probabilities = predict_proba_early_weighted_lda(
            lda=lda,
            features=features,
            lambda_early=lambda_early,
        )
        predictions = np.asarray(
            CLASS_LABELS
        )[
            np.argmax(
                probabilities,
                axis=1,
            )
        ]
        accuracy = float(
            accuracy_score(
                labels,
                predictions,
            )
        )

        if accuracy > best_accuracy:
            best_lambda = lambda_early
            best_accuracy = accuracy

    return (
        float(
            best_lambda
        ),
        float(
            best_accuracy
        ),
    )


def predict_proba_early_weighted_lda(
    lda,
    features: np.ndarray,
    lambda_early: float,
) -> np.ndarray:
    """
    Predict probabilities from LDA with weighted Early score contribution.
    """
    scores = compute_early_weighted_scores(
        lda,
        features,
        lambda_early,
    )
    probabilities = _softmax(
        scores
    )

    return _align_probabilities(
        probabilities,
        lda.classes_,
    )


def compute_early_weighted_scores(
    lda,
    features: np.ndarray,
    lambda_early: float,
) -> np.ndarray:
    """
    Compute modified LDA scores with lambda_early applied after fitting.
    """
    coefficients = np.asarray(
        lda.coef_,
        dtype=np.float64,
    )

    if coefficients.ndim == 1:
        coefficients = coefficients[
            np.newaxis,
            :
        ]

    intercept = np.asarray(
        lda.intercept_,
        dtype=np.float64,
    )

    early = _feature_contribution(
        features,
        coefficients,
        0,
        CSP_N_COMPONENTS,
    )
    middle = _feature_contribution(
        features,
        coefficients,
        CSP_N_COMPONENTS,
        2 * CSP_N_COMPONENTS,
    )
    late = _feature_contribution(
        features,
        coefficients,
        2 * CSP_N_COMPONENTS,
        3 * CSP_N_COMPONENTS,
    )

    return (
        lambda_early * early
        + middle
        + late
        + intercept
    )


def compute_accuracy_statistic(
    rows: list[SubjectEarlyWeightedResult],
) -> EarlyWeightedAccuracyStatistic:
    """
    Run the paired Wilcoxon test on subject accuracies.
    """
    baseline = np.asarray(
        [
            row.baseline_accuracy
            for row in rows
        ],
        dtype=np.float64,
    )
    early_weighted = np.asarray(
        [
            row.early_weighted_accuracy
            for row in rows
        ],
        dtype=np.float64,
    )
    finite_mask = (
        np.isfinite(
            baseline
        )
        & np.isfinite(
            early_weighted
        )
    )
    baseline = baseline[finite_mask]
    early_weighted = early_weighted[finite_mask]

    if len(
        baseline
    ) < 2:
        raise ValueError(
            "At least two paired subject accuracies are required."
        )

    statistic, p_value = wilcoxon(
        early_weighted,
        baseline,
        alternative="two-sided",
    )
    differences = (
        early_weighted
        - baseline
    )

    return EarlyWeightedAccuracyStatistic(
        n_subjects=len(
            baseline
        ),
        baseline_mean=float(
            np.mean(
                baseline
            )
        ),
        baseline_sd=float(
            np.std(
                baseline,
                ddof=1,
            )
        ),
        early_weighted_mean=float(
            np.mean(
                early_weighted
            )
        ),
        early_weighted_sd=float(
            np.std(
                early_weighted,
                ddof=1,
            )
        ),
        mean_paired_difference=float(
            np.mean(
                differences
            )
        ),
        wilcoxon_statistic=float(
            statistic
        ),
        p_value=float(
            p_value
        ),
        significant=bool(
            p_value < ALPHA
        ),
    )


def save_subject_results(
    rows: list[SubjectEarlyWeightedResult],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save subject-wise Early-weighted experiment results.
    """
    if output_file is None:
        output_file = get_csp_lda_early_weighted_subject_results_path()

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
                "subject",
                "baseline_accuracy",
                "early_weighted_accuracy",
                "accuracy_difference",
                "baseline_left_hand_recall",
                "early_weighted_left_hand_recall",
                "baseline_right_hand_recall",
                "early_weighted_right_hand_recall",
                "baseline_feet_recall",
                "early_weighted_feet_recall",
                "baseline_tongue_recall",
                "early_weighted_tongue_recall",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "subject": row.subject,
                "baseline_accuracy": _format_float(
                    row.baseline_accuracy
                ),
                "early_weighted_accuracy": _format_float(
                    row.early_weighted_accuracy
                ),
                "accuracy_difference": _format_float(
                    row.accuracy_difference
                ),
                "baseline_left_hand_recall": _format_float(
                    row.baseline_recalls["left_hand"]
                ),
                "early_weighted_left_hand_recall": _format_float(
                    row.early_weighted_recalls["left_hand"]
                ),
                "baseline_right_hand_recall": _format_float(
                    row.baseline_recalls["right_hand"]
                ),
                "early_weighted_right_hand_recall": _format_float(
                    row.early_weighted_recalls["right_hand"]
                ),
                "baseline_feet_recall": _format_float(
                    row.baseline_recalls["feet"]
                ),
                "early_weighted_feet_recall": _format_float(
                    row.early_weighted_recalls["feet"]
                ),
                "baseline_tongue_recall": _format_float(
                    row.baseline_recalls["tongue"]
                ),
                "early_weighted_tongue_recall": _format_float(
                    row.early_weighted_recalls["tongue"]
                ),
            })

    return output_file


def save_selected_lambdas(
    rows: list[SelectedLambdaRow],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save selected lambda_early values per subject/fold.
    """
    if output_file is None:
        output_file = get_csp_lda_early_weighted_selected_lambdas_path()

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
                "subject",
                "fold",
                "lambda_early",
                "validation_accuracy",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "subject": row.subject,
                "fold": row.fold,
                "lambda_early": _format_float(
                    row.lambda_early
                ),
                "validation_accuracy": _format_float(
                    row.validation_accuracy
                ),
            })

    return output_file


def save_class_summary(
    rows: list[SubjectEarlyWeightedResult],
    output_file: str | Path | None = None,
) -> Path:
    """
    Save class-wise recall summaries across subjects.
    """
    if output_file is None:
        output_file = get_csp_lda_early_weighted_class_summary_path()

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
                "class_name",
                "baseline_mean_recall",
                "baseline_sd_recall",
                "early_weighted_mean_recall",
                "early_weighted_sd_recall",
                "mean_difference",
            ],
        )
        writer.writeheader()

        for class_name, class_key in zip(
            CLASS_NAMES,
            CLASS_KEYS,
            strict=True,
        ):
            baseline_values = np.asarray(
                [
                    row.baseline_recalls[class_key]
                    for row in rows
                ],
                dtype=np.float64,
            )
            early_weighted_values = np.asarray(
                [
                    row.early_weighted_recalls[class_key]
                    for row in rows
                ],
                dtype=np.float64,
            )

            writer.writerow({
                "class_name": class_name,
                "baseline_mean_recall": _format_float(
                    np.mean(
                        baseline_values
                    )
                ),
                "baseline_sd_recall": _format_float(
                    np.std(
                        baseline_values,
                        ddof=1,
                    )
                ),
                "early_weighted_mean_recall": _format_float(
                    np.mean(
                        early_weighted_values
                    )
                ),
                "early_weighted_sd_recall": _format_float(
                    np.std(
                        early_weighted_values,
                        ddof=1,
                    )
                ),
                "mean_difference": _format_float(
                    np.mean(
                        early_weighted_values
                        - baseline_values
                    )
                ),
            })

    return output_file


def save_pooled_classification_report(
    result: EarlyWeightedExperimentResult,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save pooled baseline and Early-weighted classification reports.
    """
    if output_file is None:
        output_file = get_csp_lda_early_weighted_classification_report_path()

    output_file = Path(
        output_file
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("Baseline CSP+LDA - All subjects\n")
        file.write("=" * 70)
        file.write("\n")
        file.write(
            classification_report(
                result.y_true,
                result.baseline_predictions,
                labels=CLASS_LABELS,
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )
        file.write("\n\nEarly-weighted CSP+LDA - All subjects\n")
        file.write("=" * 70)
        file.write("\n")
        file.write(
            classification_report(
                result.y_true,
                result.early_weighted_predictions,
                labels=CLASS_LABELS,
                target_names=CLASS_NAMES,
                digits=4,
                zero_division=0,
            )
        )

    return output_file


def save_pooled_confusion_matrix(
    result: EarlyWeightedExperimentResult,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save pooled baseline and Early-weighted confusion matrices.
    """
    if output_file is None:
        output_file = get_csp_lda_early_weighted_confusion_matrix_path()

    output_file = Path(
        output_file
    )
    baseline_matrix = confusion_matrix(
        result.y_true,
        result.baseline_predictions,
        labels=CLASS_LABELS,
    )
    early_weighted_matrix = confusion_matrix(
        result.y_true,
        result.early_weighted_predictions,
        labels=CLASS_LABELS,
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
            "model",
            "true_class",
            *[
                f"predicted_{class_name}"
                for class_name in CLASS_NAMES
            ],
        ])

        for model_name, matrix in (
            ("baseline_csp_lda", baseline_matrix),
            ("early_weighted_csp_lda", early_weighted_matrix),
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


def save_accuracy_statistic(
    statistic: EarlyWeightedAccuracyStatistic,
    output_file: str | Path | None = None,
) -> Path:
    """
    Save the paired accuracy Wilcoxon result.
    """
    if output_file is None:
        output_file = get_csp_lda_early_weighted_accuracy_statistics_path()

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
                "n_subjects",
                "baseline_mean",
                "baseline_sd",
                "early_weighted_mean",
                "early_weighted_sd",
                "mean_paired_difference",
                "wilcoxon_statistic",
                "p_value",
                "alpha",
                "significant",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "comparison": "baseline CSP+LDA vs Early-weighted CSP+LDA",
            "n_subjects": statistic.n_subjects,
            "baseline_mean": _format_float(
                statistic.baseline_mean
            ),
            "baseline_sd": _format_float(
                statistic.baseline_sd
            ),
            "early_weighted_mean": _format_float(
                statistic.early_weighted_mean
            ),
            "early_weighted_sd": _format_float(
                statistic.early_weighted_sd
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
            "alpha": _format_float(
                ALPHA
            ),
            "significant": statistic.significant,
        })

    return output_file


def print_experiment_summary(
    result: EarlyWeightedExperimentResult,
    statistic: EarlyWeightedAccuracyStatistic,
) -> None:
    """
    Print subject, class-wise, pooled, and statistical summaries.
    """
    print()
    print("=" * 70)
    print("Early-weighted CSP+LDA experiment")
    print("=" * 70)
    print(
        f"{'Subject':<10} "
        f"{'Baseline':>10} "
        f"{'Early-wt':>10} "
        f"{'Delta':>10} "
        f"{'Lambdas':<24}"
    )

    lambda_lookup = {
        row.subject: []
        for row in result.selected_lambdas
    }

    for row in result.selected_lambdas:
        lambda_lookup[row.subject].append(
            row.lambda_early
        )

    for row in result.subject_results:
        lambdas = ", ".join(
            f"{value:g}"
            for value in lambda_lookup.get(
                row.subject,
                [],
            )
        )
        print(
            f"{row.subject:<10} "
            f"{row.baseline_accuracy:>10.4f} "
            f"{row.early_weighted_accuracy:>10.4f} "
            f"{row.accuracy_difference:>10.4f} "
            f"{lambdas:<24}"
        )

    print("-" * 70)
    print(
        "Baseline mean+/-SD: "
        f"{statistic.baseline_mean:.4f} +/- "
        f"{statistic.baseline_sd:.4f}"
    )
    print(
        "Early-weighted mean+/-SD: "
        f"{statistic.early_weighted_mean:.4f} +/- "
        f"{statistic.early_weighted_sd:.4f}"
    )
    print(
        "Mean paired difference: "
        f"{statistic.mean_paired_difference:.4f}"
    )
    print(
        f"Wilcoxon W={statistic.wilcoxon_statistic:.4g}, "
        f"p={statistic.p_value:.4g}, "
        f"significant={statistic.significant}"
    )

    print()
    print("Class-wise recall across subjects")
    print(
        f"{'Class':<12} "
        f"{'Baseline mean+/-SD':<22} "
        f"{'Early-wt mean+/-SD':<22} "
        f"{'Delta':>10}"
    )

    for class_name, class_key in zip(
        CLASS_NAMES,
        CLASS_KEYS,
        strict=True,
    ):
        baseline_values = np.asarray(
            [
                row.baseline_recalls[class_key]
                for row in result.subject_results
            ],
            dtype=np.float64,
        )
        early_weighted_values = np.asarray(
            [
                row.early_weighted_recalls[class_key]
                for row in result.subject_results
            ],
            dtype=np.float64,
        )
        print(
            f"{class_name:<12} "
            f"{_mean_sd(baseline_values):<22} "
            f"{_mean_sd(early_weighted_values):<22} "
            f"{np.mean(early_weighted_values - baseline_values):>10.4f}"
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
    print("Early-weighted CSP+LDA - All subjects")
    print(
        classification_report(
            result.y_true,
            result.early_weighted_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )
    print("Pooled baseline confusion matrix")
    print(
        confusion_matrix(
            result.y_true,
            result.baseline_predictions,
            labels=CLASS_LABELS,
        )
    )
    print("Pooled Early-weighted confusion matrix")
    print(
        confusion_matrix(
            result.y_true,
            result.early_weighted_predictions,
            labels=CLASS_LABELS,
        )
    )


def _feature_contribution(
    features: np.ndarray,
    coefficients: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    return (
        features[:, start:stop]
        @ coefficients[:, start:stop].T
    )


def _align_probabilities(
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros(
        (
            len(probabilities),
            len(CLASS_LABELS),
        ),
        dtype=np.float64,
    )

    for class_index, class_label in enumerate(
        classes
    ):
        output_index = CLASS_LABELS.index(
            int(
                class_label
            )
        )
        aligned[:, output_index] = probabilities[
            :,
            class_index,
        ]

    return aligned


def _softmax(
    scores: np.ndarray,
) -> np.ndarray:
    shifted_scores = (
        scores
        - np.max(
            scores,
            axis=1,
            keepdims=True,
        )
    )
    exp_scores = np.exp(
        shifted_scores
    )

    return (
        exp_scores
        / np.sum(
            exp_scores,
            axis=1,
            keepdims=True,
        )
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
            np.mean(
                y_pred[class_mask] == class_label
            )
        )

    return recalls


def _create_times(
    n_times: int,
) -> np.ndarray:
    return EPOCH_TMIN + np.arange(
        n_times
    ) / SFREQ


def _time_window_mask(
    times: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    if end == TEMPORAL_WINDOWS[-1][2]:
        return (
            (times >= start)
            & (times <= end)
        )

    return (
        (times >= start)
        & (times < end)
    )


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
    if not np.isfinite(
        value
    ):
        return "nan"

    return f"{value:.10g}"
