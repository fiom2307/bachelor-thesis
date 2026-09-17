import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from src.analysis.performance_statistical_analysis import (
    _load_subject_predictions,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_NAME_TO_LABEL
from src.data.preprocessing import BCI_2A_CHANNEL_NAMES
from src.data.preprocessing import normalize_epochs, prepare_eegnet_input
from src.models.csp_lda import (
    predict_csp_lda,
    train_or_load_csp_lda_channel_ablation,
    train_or_load_csp_lda_roi_ablation,
)
from src.models.eegnet import (
    predict_eegnet,
    train_or_load_eegnet_channel_ablation,
    train_or_load_eegnet_roi_ablation,
)
from src.utils.paths import SPATIAL_ABLATION_RESULTS_DIR, get_subject_name


MODEL_CSP_LDA = "CSP+LDA"
MODEL_EEGNET = "EEGNet"

CHANNEL_ABLATION_RESULTS_PATH = (
    SPATIAL_ABLATION_RESULTS_DIR
    / "channel_ablation_results.csv"
)
CHANNEL_ABLATION_DELTAS_PATH = (
    SPATIAL_ABLATION_RESULTS_DIR
    / "channel_ablation_deltas.csv"
)
ROI_ABLATION_RESULTS_PATH = (
    SPATIAL_ABLATION_RESULTS_DIR
    / "roi_ablation_results.csv"
)
ROI_ABLATION_DELTAS_PATH = (
    SPATIAL_ABLATION_RESULTS_DIR
    / "roi_ablation_deltas.csv"
)


@dataclass(frozen=True)
class SpatialAblationResult:
    subject: str
    model: str
    condition: str
    removed_channel: str
    removed_roi: str
    removed_channels: str
    n_channels_remaining: int
    accuracy: float
    left_hand_recall: float
    right_hand_recall: float
    feet_recall: float
    tongue_recall: float


@dataclass(frozen=True)
class SpatialAblationDelta:
    subject: str
    model: str
    condition: str
    removed_channel: str
    removed_roi: str
    removed_channels: str
    n_channels_remaining: int
    accuracy_drop: float
    left_hand_recall_drop: float
    right_hand_recall_drop: float
    feet_recall_drop: float
    tongue_recall_drop: float


def run_channel_ablation() -> None:
    baseline_rows = collect_baseline_results()
    results = list(baseline_rows)

    for removed_channel in BCI_2A_CHANNEL_NAMES:
        condition = f"without_{removed_channel}"
        print()
        print("=" * 50)
        print(f"CHANNEL ABLATION: {condition}")
        print("=" * 50)

        results.extend(
            _run_ablation_condition(
                condition=condition,
                removed_channels=[removed_channel],
                removed_channel=removed_channel,
                removed_roi="",
                ablation_kind="channel",
            )
        )

    deltas = compute_deltas(
        baseline_rows,
        results,
    )

    save_channel_results(
        results,
        CHANNEL_ABLATION_RESULTS_PATH,
    )
    save_channel_deltas(
        deltas,
        CHANNEL_ABLATION_DELTAS_PATH,
    )
    print_channel_summary(
        results,
        deltas,
    )

    print()
    print(f"Saved channel ablation results: {CHANNEL_ABLATION_RESULTS_PATH}")
    print(f"Saved channel ablation deltas: {CHANNEL_ABLATION_DELTAS_PATH}")


def run_roi_ablation() -> None:
    roi_definitions = get_roi_definitions()
    validate_roi_definitions(
        roi_definitions
    )

    baseline_rows = collect_baseline_results()
    results = list(baseline_rows)

    for roi_name, removed_channels in roi_definitions.items():
        condition = f"without_{roi_name}"
        remaining_channels = _channels_to_keep(
            removed_channels
        )

        print()
        print("=" * 50)
        print(f"ROI ABLATION: {condition}")
        print(f"Removed: {', '.join(removed_channels)}")
        print(f"Remaining channels: {len(remaining_channels)}")
        print("=" * 50)

        results.extend(
            _run_ablation_condition(
                condition=condition,
                removed_channels=removed_channels,
                removed_channel="",
                removed_roi=roi_name,
                ablation_kind="roi",
            )
        )

    deltas = compute_deltas(
        baseline_rows,
        results,
    )

    save_roi_results(
        results,
        ROI_ABLATION_RESULTS_PATH,
    )
    save_roi_deltas(
        deltas,
        ROI_ABLATION_DELTAS_PATH,
    )
    print_roi_summary(
        results,
        deltas,
    )

    print()
    print(f"Saved ROI ablation results: {ROI_ABLATION_RESULTS_PATH}")
    print(f"Saved ROI ablation deltas: {ROI_ABLATION_DELTAS_PATH}")


def collect_baseline_results() -> list[SpatialAblationResult]:
    rows = []

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )
        print(f"Baseline {subject_name}")

        try:
            y_true, csp_predictions, eegnet_predictions = (
                _load_subject_predictions(subject)
            )
        except Exception as error:
            print(f"[ERROR] Baseline {subject_name}: {error}")
            raise

        rows.append(
            _make_result_row(
                subject=subject_name,
                model=MODEL_CSP_LDA,
                condition="baseline",
                removed_channel="baseline",
                removed_roi="baseline",
                removed_channels="",
                n_channels_remaining=len(BCI_2A_CHANNEL_NAMES),
                y_true=y_true,
                predictions=csp_predictions,
            )
        )
        rows.append(
            _make_result_row(
                subject=subject_name,
                model=MODEL_EEGNET,
                condition="baseline",
                removed_channel="baseline",
                removed_roi="baseline",
                removed_channels="",
                n_channels_remaining=len(BCI_2A_CHANNEL_NAMES),
                y_true=y_true,
                predictions=eegnet_predictions,
            )
        )

    return rows


def compute_deltas(
    baseline_rows: list[SpatialAblationResult],
    results: list[SpatialAblationResult],
) -> list[SpatialAblationDelta]:
    baselines = {
        (row.subject, row.model): row
        for row in baseline_rows
    }
    deltas = []

    for row in results:
        if row.condition == "baseline":
            continue

        baseline = baselines[(row.subject, row.model)]
        deltas.append(
            SpatialAblationDelta(
                subject=row.subject,
                model=row.model,
                condition=row.condition,
                removed_channel=row.removed_channel,
                removed_roi=row.removed_roi,
                removed_channels=row.removed_channels,
                n_channels_remaining=row.n_channels_remaining,
                accuracy_drop=baseline.accuracy - row.accuracy,
                left_hand_recall_drop=(
                    baseline.left_hand_recall
                    - row.left_hand_recall
                ),
                right_hand_recall_drop=(
                    baseline.right_hand_recall
                    - row.right_hand_recall
                ),
                feet_recall_drop=baseline.feet_recall - row.feet_recall,
                tongue_recall_drop=(
                    baseline.tongue_recall
                    - row.tongue_recall
                ),
            )
        )

    return deltas


def save_channel_results(
    results: list[SpatialAblationResult],
    output_path: Path,
) -> None:
    rows = [
        {
            "subject": row.subject,
            "model": row.model,
            "removed_channel": row.removed_channel,
            "accuracy": _format_float(row.accuracy),
            "left_hand_recall": _format_float(row.left_hand_recall),
            "right_hand_recall": _format_float(row.right_hand_recall),
            "feet_recall": _format_float(row.feet_recall),
            "tongue_recall": _format_float(row.tongue_recall),
        }
        for row in results
        if row.removed_channel
    ]
    _write_csv(
        output_path,
        rows,
        [
            "subject",
            "model",
            "removed_channel",
            "accuracy",
            "left_hand_recall",
            "right_hand_recall",
            "feet_recall",
            "tongue_recall",
        ],
    )


def save_channel_deltas(
    deltas: list[SpatialAblationDelta],
    output_path: Path,
) -> None:
    rows = [
        {
            "subject": row.subject,
            "model": row.model,
            "removed_channel": row.removed_channel,
            "accuracy_drop": _format_float(row.accuracy_drop),
            "left_hand_recall_drop": _format_float(
                row.left_hand_recall_drop
            ),
            "right_hand_recall_drop": _format_float(
                row.right_hand_recall_drop
            ),
            "feet_recall_drop": _format_float(row.feet_recall_drop),
            "tongue_recall_drop": _format_float(
                row.tongue_recall_drop
            ),
        }
        for row in deltas
        if row.removed_channel
    ]
    _write_csv(
        output_path,
        rows,
        [
            "subject",
            "model",
            "removed_channel",
            "accuracy_drop",
            "left_hand_recall_drop",
            "right_hand_recall_drop",
            "feet_recall_drop",
            "tongue_recall_drop",
        ],
    )


def save_roi_results(
    results: list[SpatialAblationResult],
    output_path: Path,
) -> None:
    rows = [
        {
            "subject": row.subject,
            "model": row.model,
            "removed_roi": row.removed_roi,
            "removed_channels": row.removed_channels,
            "n_channels_remaining": row.n_channels_remaining,
            "accuracy": _format_float(row.accuracy),
            "left_hand_recall": _format_float(row.left_hand_recall),
            "right_hand_recall": _format_float(row.right_hand_recall),
            "feet_recall": _format_float(row.feet_recall),
            "tongue_recall": _format_float(row.tongue_recall),
        }
        for row in results
        if row.removed_roi
    ]
    _write_csv(
        output_path,
        rows,
        [
            "subject",
            "model",
            "removed_roi",
            "removed_channels",
            "n_channels_remaining",
            "accuracy",
            "left_hand_recall",
            "right_hand_recall",
            "feet_recall",
            "tongue_recall",
        ],
    )


def save_roi_deltas(
    deltas: list[SpatialAblationDelta],
    output_path: Path,
) -> None:
    rows = [
        {
            "subject": row.subject,
            "model": row.model,
            "removed_roi": row.removed_roi,
            "removed_channels": row.removed_channels,
            "n_channels_remaining": row.n_channels_remaining,
            "accuracy_drop": _format_float(row.accuracy_drop),
            "left_hand_recall_drop": _format_float(
                row.left_hand_recall_drop
            ),
            "right_hand_recall_drop": _format_float(
                row.right_hand_recall_drop
            ),
            "feet_recall_drop": _format_float(row.feet_recall_drop),
            "tongue_recall_drop": _format_float(
                row.tongue_recall_drop
            ),
        }
        for row in deltas
        if row.removed_roi
    ]
    _write_csv(
        output_path,
        rows,
        [
            "subject",
            "model",
            "removed_roi",
            "removed_channels",
            "n_channels_remaining",
            "accuracy_drop",
            "left_hand_recall_drop",
            "right_hand_recall_drop",
            "feet_recall_drop",
            "tongue_recall_drop",
        ],
    )


def print_channel_summary(
    results: list[SpatialAblationResult],
    deltas: list[SpatialAblationDelta],
) -> None:
    delta_lookup = _mean_delta_lookup(deltas)

    print()
    print(
        "Model       Removed Channel  Accuracy  Accuracy Drop  "
        "RH Recall  RH Recall Drop"
    )
    print("-" * 80)

    for model in (MODEL_CSP_LDA, MODEL_EEGNET):
        for removed_channel in ["baseline", *BCI_2A_CHANNEL_NAMES]:
            rows = [
                row
                for row in results
                if row.model == model
                and row.removed_channel == removed_channel
            ]
            if not rows:
                continue

            delta = delta_lookup.get(
                (model, f"without_{removed_channel}")
            )
            accuracy_drop = "-" if delta is None else _format_float(
                delta.accuracy_drop
            )
            right_hand_drop = "-" if delta is None else _format_float(
                delta.right_hand_recall_drop
            )

            print(
                f"{model:<11}"
                f"{removed_channel:<17}"
                f"{_mean([row.accuracy for row in rows]):>8.4f}  "
                f"{accuracy_drop:>13}  "
                f"{_mean([row.right_hand_recall for row in rows]):>9.4f}  "
                f"{right_hand_drop:>14}"
            )


def print_roi_summary(
    results: list[SpatialAblationResult],
    deltas: list[SpatialAblationDelta],
) -> None:
    delta_lookup = _mean_delta_lookup(deltas)
    roi_names = [
        "baseline",
        *get_roi_definitions().keys(),
    ]

    print()
    print(
        "Model       Removed ROI        Accuracy  Accuracy Drop  "
        "LH Recall  LH Drop  RH Recall  RH Drop  "
        "Feet Recall  Feet Drop  Tongue Recall  Tongue Drop"
    )
    print("-" * 145)

    for model in (MODEL_CSP_LDA, MODEL_EEGNET):
        for removed_roi in roi_names:
            rows = [
                row
                for row in results
                if row.model == model and row.removed_roi == removed_roi
            ]
            if not rows:
                continue

            delta = delta_lookup.get(
                (model, f"without_{removed_roi}")
            )
            accuracy_drop = "-" if delta is None else _format_float(
                delta.accuracy_drop
            )
            lh_drop = "-" if delta is None else _format_float(
                delta.left_hand_recall_drop
            )
            rh_drop = "-" if delta is None else _format_float(
                delta.right_hand_recall_drop
            )
            feet_drop = "-" if delta is None else _format_float(
                delta.feet_recall_drop
            )
            tongue_drop = "-" if delta is None else _format_float(
                delta.tongue_recall_drop
            )

            print(
                f"{model:<11}"
                f"{removed_roi:<19}"
                f"{_mean([row.accuracy for row in rows]):>8.4f}  "
                f"{accuracy_drop:>13}  "
                f"{_mean([row.left_hand_recall for row in rows]):>9.4f}  "
                f"{lh_drop:>7}  "
                f"{_mean([row.right_hand_recall for row in rows]):>9.4f}  "
                f"{rh_drop:>7}  "
                f"{_mean([row.feet_recall for row in rows]):>11.4f}  "
                f"{feet_drop:>9}  "
                f"{_mean([row.tongue_recall for row in rows]):>13.4f}  "
                f"{tongue_drop:>11}"
            )


def get_roi_definitions() -> dict[str, list[str]]:
    return {
        "left_sensorimotor": [
            "FC3",
            "FC1",
            "C5",
            "C3",
            "C1",
            "CP3",
            "CP1",
            "P1",
        ],
        "midline": [
            "Fz",
            "FCz",
            "Cz",
            "CPz",
            "Pz",
            "POz",
        ],
        "right_sensorimotor": [
            "FC2",
            "FC4",
            "C2",
            "C4",
            "C6",
            "CP2",
            "CP4",
            "P2",
        ],
    }


def validate_roi_definitions(
    roi_definitions: dict[str, list[str]],
) -> None:
    channel_sets = [
        set(channels)
        for channels in roi_definitions.values()
    ]
    all_roi_channels = set().union(
        *channel_sets
    )

    for index, channel_set in enumerate(channel_sets):
        for other_set in channel_sets[index + 1:]:
            if channel_set & other_set:
                raise ValueError(
                    "ROI channel definitions are not disjoint."
                )

    expected_channels = set(BCI_2A_CHANNEL_NAMES)
    if all_roi_channels != expected_channels:
        missing = sorted(
            expected_channels - all_roi_channels
        )
        extra = sorted(
            all_roi_channels - expected_channels
        )
        raise ValueError(
            f"ROI definitions do not cover exactly the 22 EEG channels. "
            f"Missing={missing}; extra={extra}."
        )


def _run_ablation_condition(
    condition: str,
    removed_channels: list[str],
    removed_channel: str,
    removed_roi: str,
    ablation_kind: str,
) -> list[SpatialAblationResult]:
    rows = []
    removed_channels_text = ";".join(removed_channels)
    keep_indices = _channel_indices(
        _channels_to_keep(removed_channels)
    )

    for subject in range(1, 10):
        subject_name = get_subject_name(
            subject
        )
        print(f"Subject {subject_name}")

        try:
            data = get_data_for_subject(subject)
            if data is None:
                raise FileNotFoundError(
                    f"Data for {subject_name} not found."
                )

            X_train, y_train, X_eval, y_eval = data
            X_train = X_train[:, keep_indices, :]
            X_eval = X_eval[:, keep_indices, :]

            rows.append(
                _evaluate_csp_lda_ablation(
                    subject_name=subject_name,
                    subject=subject,
                    condition=condition,
                    removed_channel=removed_channel,
                    removed_roi=removed_roi,
                    removed_channels=removed_channels_text,
                    ablation_kind=ablation_kind,
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_eval,
                    y_eval=y_eval,
                )
            )
            rows.append(
                _evaluate_eegnet_ablation(
                    subject_name=subject_name,
                    subject=subject,
                    condition=condition,
                    removed_channel=removed_channel,
                    removed_roi=removed_roi,
                    removed_channels=removed_channels_text,
                    ablation_kind=ablation_kind,
                    X_train=X_train,
                    y_train=y_train,
                    X_eval=X_eval,
                    y_eval=y_eval,
                )
            )
        except Exception as error:
            print(f"[ERROR] {condition} {subject_name}: {error}")
            raise

    return rows


def _evaluate_csp_lda_ablation(
    subject_name: str,
    subject: int,
    condition: str,
    removed_channel: str,
    removed_roi: str,
    removed_channels: str,
    ablation_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> SpatialAblationResult:
    print("  CSP+LDA ...")

    if ablation_kind == "channel":
        models = train_or_load_csp_lda_channel_ablation(
            subject,
            X_train,
            y_train,
            condition,
        )
    elif ablation_kind == "roi":
        models = train_or_load_csp_lda_roi_ablation(
            subject,
            X_train,
            y_train,
            condition,
        )
    else:
        raise ValueError(
            f"Unknown ablation kind: {ablation_kind}"
        )

    predictions = predict_csp_lda(
        models,
        X_eval,
    )

    return _make_result_row(
        subject=subject_name,
        model=MODEL_CSP_LDA,
        condition=condition,
        removed_channel=removed_channel,
        removed_roi=removed_roi,
        removed_channels=removed_channels,
        n_channels_remaining=X_train.shape[1],
        y_true=y_eval,
        predictions=predictions,
    )


def _evaluate_eegnet_ablation(
    subject_name: str,
    subject: int,
    condition: str,
    removed_channel: str,
    removed_roi: str,
    removed_channels: str,
    ablation_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> SpatialAblationResult:
    print("  EEGNet ...")

    X_train, X_eval = normalize_epochs(
        X_train,
        X_eval,
    )
    X_train = prepare_eegnet_input(
        X_train
    )
    X_eval = prepare_eegnet_input(
        X_eval
    )

    if ablation_kind == "channel":
        models = train_or_load_eegnet_channel_ablation(
            subject,
            X_train,
            y_train,
            condition,
        )
    elif ablation_kind == "roi":
        models = train_or_load_eegnet_roi_ablation(
            subject,
            X_train,
            y_train,
            condition,
        )
    else:
        raise ValueError(
            f"Unknown ablation kind: {ablation_kind}"
        )

    predictions = predict_eegnet(
        models,
        X_eval,
    )

    return _make_result_row(
        subject=subject_name,
        model=MODEL_EEGNET,
        condition=condition,
        removed_channel=removed_channel,
        removed_roi=removed_roi,
        removed_channels=removed_channels,
        n_channels_remaining=X_train.shape[1],
        y_true=y_eval,
        predictions=predictions,
    )


def _make_result_row(
    subject: str,
    model: str,
    condition: str,
    removed_channel: str,
    removed_roi: str,
    removed_channels: str,
    n_channels_remaining: int,
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> SpatialAblationResult:
    recalls = {
        class_name: _class_recall(
            y_true,
            predictions,
            class_label,
        )
        for class_name, class_label in CLASS_NAME_TO_LABEL.items()
    }

    return SpatialAblationResult(
        subject=subject,
        model=model,
        condition=condition,
        removed_channel=removed_channel,
        removed_roi=removed_roi,
        removed_channels=removed_channels,
        n_channels_remaining=n_channels_remaining,
        accuracy=float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
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
    y_true = np.asarray(
        y_true
    )
    predictions = np.asarray(
        predictions
    )
    class_mask = y_true == class_label

    if not np.any(class_mask):
        return np.nan

    return float(
        np.mean(
            predictions[class_mask] == class_label
        )
    )


def _channels_to_keep(
    channels_to_remove: list[str],
) -> list[str]:
    unknown_channels = [
        channel
        for channel in channels_to_remove
        if channel not in BCI_2A_CHANNEL_NAMES
    ]

    if unknown_channels:
        raise ValueError(
            f"Unknown EEG channels: {unknown_channels}"
        )

    return [
        channel
        for channel in BCI_2A_CHANNEL_NAMES
        if channel not in channels_to_remove
    ]


def _channel_indices(
    channels_to_keep: list[str],
) -> list[int]:
    return [
        BCI_2A_CHANNEL_NAMES.index(channel)
        for channel in channels_to_keep
    ]


def _mean_delta_lookup(
    deltas: list[SpatialAblationDelta],
) -> dict[tuple[str, str], SpatialAblationDelta]:
    lookup = {}

    for model in (MODEL_CSP_LDA, MODEL_EEGNET):
        conditions = sorted(
            {
                row.condition
                for row in deltas
                if row.model == model
            }
        )

        for condition in conditions:
            rows = [
                row
                for row in deltas
                if row.model == model and row.condition == condition
            ]
            lookup[(model, condition)] = SpatialAblationDelta(
                subject="mean",
                model=model,
                condition=condition,
                removed_channel=rows[0].removed_channel,
                removed_roi=rows[0].removed_roi,
                removed_channels=rows[0].removed_channels,
                n_channels_remaining=rows[0].n_channels_remaining,
                accuracy_drop=_mean(
                    [row.accuracy_drop for row in rows]
                ),
                left_hand_recall_drop=_mean(
                    [row.left_hand_recall_drop for row in rows]
                ),
                right_hand_recall_drop=_mean(
                    [row.right_hand_recall_drop for row in rows]
                ),
                feet_recall_drop=_mean(
                    [row.feet_recall_drop for row in rows]
                ),
                tongue_recall_drop=_mean(
                    [row.tongue_recall_drop for row in rows]
                ),
            )

    return lookup


def _write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
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
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


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


def _format_float(
    value: float,
) -> str:
    if not np.isfinite(value):
        return ""

    return f"{value:.10g}"
