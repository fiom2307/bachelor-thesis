import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR),
    )

from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_LABELS
from src.pipelines.comparison_pipeline import evaluate_models_for_subject
from src.pipelines.csp_lda_pipeline import (
    evaluate_csp_lda_time_window_for_subject,
)
from src.pipelines.eegnet_pipeline import (
    evaluate_eegnet_time_window_for_subject,
)
from src.utils.config import BASE_SEED
from src.utils.paths import ACCURACY_RESULTS_DIR, get_subject_name


MODEL_CSP_LDA = "CSP+LDA"
MODEL_EEGNET = "EEGNet"

CLASS_COLUMNS = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}

# The three temporal conditions are the original full-window baseline,
# removal of early information, and removal of late information.
CONDITIONS = {
    "baseline": (0.5, 4.0),
    "no_early": (1.5, 4.0),
    "no_late": (0.5, 2.5),
}

RESULTS_PATH = (
    ACCURACY_RESULTS_DIR
    / f"seed_{BASE_SEED}_temporal_ablation_results.csv"
)
DELTAS_PATH = (
    ACCURACY_RESULTS_DIR
    / f"seed_{BASE_SEED}_temporal_ablation_deltas.csv"
)


@dataclass(frozen=True)
class TemporalAblationResult:
    subject: str
    model: str
    condition: str
    tmin: float
    tmax: float
    accuracy: float
    left_hand_recall: float
    right_hand_recall: float
    feet_recall: float
    tongue_recall: float


@dataclass(frozen=True)
class TemporalAblationDelta:
    subject: str
    model: str
    condition: str
    tmin: float
    tmax: float
    accuracy_drop: float
    right_hand_recall_drop: float


def main() -> None:
    results = run_temporal_ablation_experiment()
    deltas = compute_temporal_ablation_deltas(
        results
    )

    save_results(
        results,
        RESULTS_PATH,
    )
    save_deltas(
        deltas,
        DELTAS_PATH,
    )
    print_summary(
        results,
        deltas,
    )

    print()
    print(f"Saved temporal ablation results: {RESULTS_PATH}")
    print(f"Saved temporal ablation deltas: {DELTAS_PATH}")


def run_temporal_ablation_experiment() -> list[TemporalAblationResult]:
    results = []

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )
        print(f"\nRunning {subject_name}...")

        baseline_rows = _evaluate_baseline_condition(
            subject
        )
        ablation_rows = _evaluate_ablation_conditions(
            subject
        )

        results.extend(
            baseline_rows
        )
        results.extend(
            ablation_rows
        )

    if not results:
        raise RuntimeError(
            "No temporal ablation results were generated."
        )

    return results


def _evaluate_baseline_condition(
    subject: int,
) -> list[TemporalAblationResult]:
    evaluation = evaluate_models_for_subject(
        subject
    )

    if evaluation is None:
        print(
            f"Skipping {get_subject_name(subject)} baseline: data not found"
        )
        return []

    (
        y_true,
        csp_accuracy,
        csp_predictions,
        eegnet_accuracy,
        eegnet_predictions,
    ) = evaluation

    tmin, tmax = CONDITIONS["baseline"]

    return [
        _make_result_row(
            subject=subject,
            model=MODEL_CSP_LDA,
            condition="baseline",
            tmin=tmin,
            tmax=tmax,
            accuracy=csp_accuracy,
            y_true=y_true,
            predictions=csp_predictions,
        ),
        _make_result_row(
            subject=subject,
            model=MODEL_EEGNET,
            condition="baseline",
            tmin=tmin,
            tmax=tmax,
            accuracy=eegnet_accuracy,
            y_true=y_true,
            predictions=eegnet_predictions,
        ),
    ]


def _evaluate_ablation_conditions(
    subject: int,
) -> list[TemporalAblationResult]:
    results = []

    for condition in ("no_early", "no_late"):
        tmin, tmax = CONDITIONS[condition]
        data = get_data_for_subject(
            subject,
            tmin,
            tmax,
        )

        if data is None:
            print(
                f"Skipping {get_subject_name(subject)} {condition}: "
                "data not found"
            )
            continue

        _, _, _, y_true = data

        csp_result = evaluate_csp_lda_time_window_for_subject(
            subject,
            tmin,
            tmax,
        )
        eegnet_result = evaluate_eegnet_time_window_for_subject(
            subject,
            tmin,
            tmax,
        )

        if csp_result is not None:
            csp_accuracy, csp_predictions = csp_result
            results.append(
                _make_result_row(
                    subject=subject,
                    model=MODEL_CSP_LDA,
                    condition=condition,
                    tmin=tmin,
                    tmax=tmax,
                    accuracy=csp_accuracy,
                    y_true=y_true,
                    predictions=csp_predictions,
                )
            )

        if eegnet_result is not None:
            eegnet_accuracy, eegnet_predictions = eegnet_result
            results.append(
                _make_result_row(
                    subject=subject,
                    model=MODEL_EEGNET,
                    condition=condition,
                    tmin=tmin,
                    tmax=tmax,
                    accuracy=eegnet_accuracy,
                    y_true=y_true,
                    predictions=eegnet_predictions,
                )
            )

    return results


def compute_temporal_ablation_deltas(
    results: list[TemporalAblationResult],
) -> list[TemporalAblationDelta]:
    by_subject_model_condition = {
        (row.subject, row.model, row.condition): row
        for row in results
    }
    deltas = []

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )
        for model in (MODEL_CSP_LDA, MODEL_EEGNET):
            baseline = by_subject_model_condition.get(
                (subject_name, model, "baseline")
            )

            if baseline is None:
                continue

            for condition in ("no_early", "no_late"):
                ablation = by_subject_model_condition.get(
                    (subject_name, model, condition)
                )

                if ablation is None:
                    continue

                deltas.append(
                    TemporalAblationDelta(
                        subject=subject_name,
                        model=model,
                        condition=condition,
                        tmin=ablation.tmin,
                        tmax=ablation.tmax,
                        accuracy_drop=(
                            baseline.accuracy
                            - ablation.accuracy
                        ),
                        right_hand_recall_drop=(
                            baseline.right_hand_recall
                            - ablation.right_hand_recall
                        ),
                    )
                )

    return deltas


def save_results(
    results: list[TemporalAblationResult],
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
                "model",
                "condition",
                "tmin",
                "tmax",
                "accuracy",
                "left_hand_recall",
                "right_hand_recall",
                "feet_recall",
                "tongue_recall",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                "subject": row.subject,
                "model": row.model,
                "condition": row.condition,
                "tmin": _format_float(row.tmin),
                "tmax": _format_float(row.tmax),
                "accuracy": _format_float(row.accuracy),
                "left_hand_recall": _format_float(row.left_hand_recall),
                "right_hand_recall": _format_float(row.right_hand_recall),
                "feet_recall": _format_float(row.feet_recall),
                "tongue_recall": _format_float(row.tongue_recall),
            })


def save_deltas(
    deltas: list[TemporalAblationDelta],
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
                "model",
                "condition",
                "tmin",
                "tmax",
                "accuracy_drop",
                "right_hand_recall_drop",
            ],
        )
        writer.writeheader()

        for row in deltas:
            writer.writerow({
                "subject": row.subject,
                "model": row.model,
                "condition": row.condition,
                "tmin": _format_float(row.tmin),
                "tmax": _format_float(row.tmax),
                "accuracy_drop": _format_float(row.accuracy_drop),
                "right_hand_recall_drop": _format_float(
                    row.right_hand_recall_drop
                ),
            })


def print_summary(
    results: list[TemporalAblationResult],
    deltas: list[TemporalAblationDelta],
) -> None:
    delta_by_model_condition = {
        (row.model, row.condition): row
        for row in _mean_deltas_by_model_condition(deltas)
    }

    print()
    print(
        "Model       Condition    Accuracy    RH Recall    "
        "Accuracy Drop    RH Recall Drop"
    )
    print("-" * 75)

    for model in (MODEL_CSP_LDA, MODEL_EEGNET):
        for condition in ("baseline", "no_early", "no_late"):
            condition_rows = [
                row
                for row in results
                if row.model == model and row.condition == condition
            ]

            if not condition_rows:
                continue

            accuracy = _mean(
                [row.accuracy for row in condition_rows]
            )
            right_hand_recall = _mean(
                [row.right_hand_recall for row in condition_rows]
            )
            delta = delta_by_model_condition.get(
                (model, condition)
            )
            accuracy_drop = (
                "-"
                if delta is None
                else _format_float(delta.accuracy_drop)
            )
            right_hand_recall_drop = (
                "-"
                if delta is None
                else _format_float(delta.right_hand_recall_drop)
            )

            print(
                f"{model:<11}"
                f"{_format_condition(condition):<13}"
                f"{accuracy:>8.4f}    "
                f"{right_hand_recall:>8.4f}    "
                f"{accuracy_drop:>13}    "
                f"{right_hand_recall_drop:>14}"
            )


def _mean_deltas_by_model_condition(
    deltas: list[TemporalAblationDelta],
) -> list[TemporalAblationDelta]:
    mean_rows = []

    for model in (MODEL_CSP_LDA, MODEL_EEGNET):
        for condition in ("no_early", "no_late"):
            rows = [
                row
                for row in deltas
                if row.model == model and row.condition == condition
            ]

            if not rows:
                continue

            mean_rows.append(
                TemporalAblationDelta(
                    subject="mean",
                    model=model,
                    condition=condition,
                    tmin=rows[0].tmin,
                    tmax=rows[0].tmax,
                    accuracy_drop=_mean(
                        [row.accuracy_drop for row in rows]
                    ),
                    right_hand_recall_drop=_mean(
                        [row.right_hand_recall_drop for row in rows]
                    ),
                )
            )

    return mean_rows


def _make_result_row(
    subject: int,
    model: str,
    condition: str,
    tmin: float,
    tmax: float,
    accuracy: float,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> TemporalAblationResult:
    recalls = {
        class_name: _class_recall(
            y_true,
            predictions,
            class_label,
        )
        for class_name, class_label in CLASS_COLUMNS.items()
    }

    return TemporalAblationResult(
        subject=get_subject_name(subject),
        model=model,
        condition=condition,
        tmin=tmin,
        tmax=tmax,
        accuracy=float(accuracy),
        left_hand_recall=recalls["left_hand"],
        right_hand_recall=recalls["right_hand"],
        feet_recall=recalls["feet"],
        tongue_recall=recalls["tongue"],
    )


def _class_recall(
    y_true: np.ndarray,
    predictions: np.ndarray,
    class_label: int,
) -> float:
    if class_label not in CLASS_LABELS:
        raise ValueError(
            f"Unknown class label: {class_label}"
        )

    y_true = np.asarray(
        y_true
    )
    predictions = np.asarray(
        predictions
    )
    class_mask = (
        y_true
        == class_label
    )
    n_class_trials = int(
        np.sum(class_mask)
    )

    if n_class_trials == 0:
        return np.nan

    return float(
        np.mean(
            predictions[class_mask]
            == class_label
        )
    )


def _mean(
    values: list[float],
) -> float:
    return float(
        np.nanmean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def _format_condition(
    condition: str,
) -> str:
    return condition.replace(
        "_",
        " ",
    ).title()


def _format_float(
    value: float,
) -> str:
    if not np.isfinite(value):
        return ""

    return f"{value:.10g}"


if __name__ == "__main__":
    main()
