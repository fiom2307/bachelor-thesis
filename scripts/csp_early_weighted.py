import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.data_loader import load_raw_gdf
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_NAME_TO_LABEL
from src.models.csp import apply_csp
from src.models.csp_lda import predict_csp_lda, train_csp_lda
from src.utils.config import BASE_SEED, CSP_N_COMPONENTS, EPOCH_TMIN, N_FOLDS
from src.utils.cross_validation import get_stratified_folds
from src.utils.paths import (
    ACCURACY_RESULTS_DIR,
    RESULTS_DIR,
    get_experiment_model_dir,
    get_subject_files,
    get_subject_name,
)


EXPERIMENT_NAME = "csp_early_weighted"
MODEL_NAME = "csp_lda_weighted"
EARLY_START = 0.5
EARLY_END = 1.5
EARLY_WEIGHT = 2.0
EEGNET_MEAN_ACCURACY = 0.7054
REFERENCE_BASELINE_MEAN = 0.6209
RIGHT_HAND_LABEL = CLASS_NAME_TO_LABEL["right_hand"]
RECALL_COLUMNS = {
    "left_hand": "left_hand_recall",
    "right_hand": "right_hand_recall",
    "feet": "feet_recall",
    "tongue": "tongue_recall",
}
CLASS_DISPLAY_NAMES = {
    "left_hand": "Left-Hand",
    "right_hand": "Right-Hand",
    "feet": "Feet",
    "tongue": "Tongue",
}

RESULTS_PATH = (
    RESULTS_DIR
    / "temporal_ablation"
    / "csp_early_weighted_results.csv"
)
BASELINE_RESULTS_PATH = (
    ACCURACY_RESULTS_DIR
    / f"seed_{BASE_SEED}_csp_lda_vs_eegnet.csv"
)
BASELINE_RECALL_RESULTS_PATH = (
    ACCURACY_RESULTS_DIR
    / f"seed_{BASE_SEED}_temporal_ablation_results.csv"
)
MODEL_DIR = ROOT_DIR / "models" / MODEL_NAME
LEGACY_MODEL_DIR = get_experiment_model_dir(EXPERIMENT_NAME)


@dataclass(frozen=True)
class EarlyWeightedResult:
    subject: str
    baseline_accuracy: float
    early_weighted_accuracy: float
    accuracy_difference: float
    baseline_recalls: dict[str, float]
    early_weighted_recalls: dict[str, float]
    recall_differences: dict[str, float]
    baseline_rh_recall: float
    early_weighted_rh_recall: float
    rh_recall_difference: float
    baseline_eegnet_gap: float
    weighted_eegnet_gap: float


def main() -> None:
    baseline_metrics = load_baseline_metrics(
        BASELINE_RESULTS_PATH
    )

    results = []

    for subject in range(1, 10):
        subject_name = get_subject_name(subject)
        print()
        print(f"Running {subject_name}...")

        result = evaluate_subject(
            subject,
            baseline_metrics[subject_name],
        )
        results.append(result)

        print(
            f"{subject_name}: baseline={result.baseline_accuracy:.4f}, "
            f"early-weighted={result.early_weighted_accuracy:.4f}, "
            f"difference={result.accuracy_difference:+.4f}, "
            f"baseline RH={result.baseline_rh_recall:.4f}, "
            f"early-weighted RH={result.early_weighted_rh_recall:.4f}, "
            f"RH difference={result.rh_recall_difference:+.4f}"
        )

    save_results(results, RESULTS_PATH)
    print_summary(results)
    print()
    print(f"Saved results: {RESULTS_PATH}")


def evaluate_subject(
    subject: int,
    baseline_metrics: dict[str, float],
) -> EarlyWeightedResult:
    data = get_data_for_subject(subject)

    if data is None:
        raise RuntimeError(
            f"Could not load data for {get_subject_name(subject)}."
        )

    X_train, y_train, X_eval, y_eval = data
    sfreq = get_sampling_frequency(subject)
    early_mask, temporal_scaling = make_temporal_scaling(
        n_times=X_train.shape[-1],
        sfreq=sfreq,
    )

    print_sanity_checks(
        X_train,
        temporal_scaling,
        early_mask,
        sfreq,
    )

    models = train_or_load_early_weighted_csp_lda(
        subject,
        X_train,
        y_train,
        temporal_scaling,
    )
    X_eval_weighted = apply_temporal_scaling(
        X_eval,
        temporal_scaling,
    )
    print_csp_feature_count(
        models,
        X_eval_weighted,
    )
    predictions = predict_csp_lda(
        models,
        X_eval_weighted,
    )
    accuracy = float(
        accuracy_score(
            y_eval,
            predictions,
        )
    )
    baseline_accuracy = baseline_metrics["accuracy"]
    baseline_recalls = {
        class_name: baseline_metrics[recall_column]
        for class_name, recall_column in RECALL_COLUMNS.items()
    }
    early_weighted_recalls = {
        class_name: class_recall(
            y_true=y_eval,
            predictions=predictions,
            class_label=class_label,
        )
        for class_name, class_label in CLASS_NAME_TO_LABEL.items()
    }
    recall_differences = {
        class_name: (
            early_weighted_recalls[class_name]
            - baseline_recalls[class_name]
        )
        for class_name in RECALL_COLUMNS
    }
    baseline_rh_recall = baseline_recalls["right_hand"]
    early_weighted_rh_recall = early_weighted_recalls["right_hand"]

    return EarlyWeightedResult(
        subject=get_subject_name(subject),
        baseline_accuracy=baseline_accuracy,
        early_weighted_accuracy=accuracy,
        accuracy_difference=accuracy - baseline_accuracy,
        baseline_recalls=baseline_recalls,
        early_weighted_recalls=early_weighted_recalls,
        recall_differences=recall_differences,
        baseline_rh_recall=baseline_rh_recall,
        early_weighted_rh_recall=early_weighted_rh_recall,
        rh_recall_difference=early_weighted_rh_recall - baseline_rh_recall,
        baseline_eegnet_gap=EEGNET_MEAN_ACCURACY - baseline_accuracy,
        weighted_eegnet_gap=EEGNET_MEAN_ACCURACY - accuracy,
    )


def train_or_load_early_weighted_csp_lda(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    temporal_scaling: np.ndarray,
) -> list:
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        copy_legacy_fold_models_if_available(subject, fold)
        saved_models = load_fold_models(subject, fold)

        if saved_models is not None:
            print(
                f"[LOAD] {get_subject_name(subject)} "
                f"Early-weighted CSP+LDA fold {fold}/{N_FOLDS}"
            )
            models.append(saved_models)
            continue

        print(
            f"[TRAIN] {get_subject_name(subject)} "
            f"Early-weighted CSP+LDA fold {fold}/{N_FOLDS}"
        )

        X_tr = apply_temporal_scaling(
            X_train[train_idx],
            temporal_scaling,
        )
        y_tr = y_train[train_idx]

        csp, lda = train_csp_lda(
            X_tr,
            y_tr,
            n_components=CSP_N_COMPONENTS,
        )
        check_csp_feature_count(csp, X_tr)
        save_fold_models(
            subject,
            fold,
            csp,
            lda,
        )
        models.append((csp, lda))

    return models


def make_temporal_scaling(
    n_times: int,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = EPOCH_TMIN + np.arange(n_times, dtype=np.float64) / sfreq
    early_mask = (
        (times >= EARLY_START)
        & (times < EARLY_END)
    )

    if not np.any(early_mask):
        raise RuntimeError("The Early temporal mask is empty.")

    sample_weights = np.ones(
        n_times,
        dtype=np.float64,
    )
    sample_weights[early_mask] = EARLY_WEIGHT

    return early_mask, np.sqrt(sample_weights)


def apply_temporal_scaling(
    X: np.ndarray,
    temporal_scaling: np.ndarray,
) -> np.ndarray:
    weighted = X * temporal_scaling[np.newaxis, np.newaxis, :]

    if weighted.shape != X.shape:
        raise RuntimeError(
            f"Temporal weighting changed shape from {X.shape} "
            f"to {weighted.shape}."
        )

    return weighted


def print_sanity_checks(
    X: np.ndarray,
    temporal_scaling: np.ndarray,
    early_mask: np.ndarray,
    sfreq: float,
) -> None:
    weighted_shape = apply_temporal_scaling(
        X,
        temporal_scaling,
    ).shape
    times = EPOCH_TMIN + np.arange(X.shape[-1], dtype=np.float64) / sfreq
    early_times = times[early_mask]

    print(f"original shape = {X.shape}")
    print(f"weighted shape = {weighted_shape}")
    print(f"sampling frequency = {sfreq:.6g} Hz")
    print(f"number of samples in full epoch = {X.shape[-1]}")
    print(f"number of samples classified as Early = {int(early_mask.sum())}")
    print(
        "actual Early mask times = "
        f"{early_times[0]:.6f}-{early_times[-1]:.6f} s"
    )
    print(
        "unique temporal scaling factors = "
        f"{np.unique(temporal_scaling)}"
    )


def check_csp_feature_count(
    csp,
    X: np.ndarray,
) -> None:
    n_features = apply_csp(
        csp,
        X[:1],
    ).shape[1]

    if n_features != CSP_N_COMPONENTS:
        raise RuntimeError(
            f"Expected {CSP_N_COMPONENTS} CSP features, "
            f"but found {n_features}."
        )


def print_csp_feature_count(
    models: list,
    X: np.ndarray,
) -> None:
    if not models:
        raise RuntimeError("No CSP+LDA models were available.")

    n_features = apply_csp(
        models[0][0],
        X[:1],
    ).shape[1]

    if n_features != CSP_N_COMPONENTS:
        raise RuntimeError(
            f"Expected {CSP_N_COMPONENTS} CSP features, "
            f"but found {n_features}."
        )

    print(f"CSP features per trial = {n_features}")


def get_sampling_frequency(subject: int) -> float:
    files = get_subject_files(subject)

    if files is None:
        raise RuntimeError(
            f"Could not find data files for {get_subject_name(subject)}."
        )

    train_file, _, _ = files
    raw = load_raw_gdf(train_file)

    return float(raw.info["sfreq"])


def get_fold_model_paths(
    subject: int,
    fold: int,
) -> tuple[Path, Path]:
    subject_name = get_subject_name(subject)
    subject_dir = MODEL_DIR / subject_name
    subject_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csp_path = (
        subject_dir
        / f"{subject_name}_csp_kfold_seed{BASE_SEED}_fold{fold}.joblib"
    )
    lda_path = (
        subject_dir
        / f"{subject_name}_lda_kfold_seed{BASE_SEED}_fold{fold}.joblib"
    )

    return csp_path, lda_path


def get_legacy_fold_model_paths(
    subject: int,
    fold: int,
) -> tuple[Path, Path]:
    subject_name = get_subject_name(subject)
    subject_dir = LEGACY_MODEL_DIR / subject_name

    csp_path = (
        subject_dir
        / f"{subject_name}_early_weighted_csp_seed{BASE_SEED}_fold{fold}.joblib"
    )
    lda_path = (
        subject_dir
        / f"{subject_name}_early_weighted_lda_seed{BASE_SEED}_fold{fold}.joblib"
    )

    return csp_path, lda_path


def load_fold_models(
    subject: int,
    fold: int,
):
    csp_path, lda_path = get_fold_model_paths(
        subject,
        fold,
    )

    if not (
        csp_path.exists()
        and lda_path.exists()
    ):
        return None

    return joblib.load(csp_path), joblib.load(lda_path)


def copy_legacy_fold_models_if_available(
    subject: int,
    fold: int,
) -> bool:
    csp_path, lda_path = get_fold_model_paths(
        subject,
        fold,
    )
    legacy_csp_path, legacy_lda_path = get_legacy_fold_model_paths(
        subject,
        fold,
    )

    if csp_path.exists() and lda_path.exists():
        return True

    if not (
        legacy_csp_path.exists()
        and legacy_lda_path.exists()
    ):
        return False

    if csp_path.exists() != lda_path.exists():
        raise RuntimeError(
            "Only one weighted model file exists for "
            f"{get_subject_name(subject)} fold {fold}; refusing to mix "
            "model caches."
        )

    shutil.copy2(
        legacy_csp_path,
        csp_path,
    )
    shutil.copy2(
        legacy_lda_path,
        lda_path,
    )
    print(
        f"[COPY] {get_subject_name(subject)} "
        f"Early-weighted CSP+LDA fold {fold}/{N_FOLDS} "
        f"to {MODEL_DIR}"
    )

    return True


def save_fold_models(
    subject: int,
    fold: int,
    csp,
    lda,
) -> None:
    csp_path, lda_path = get_fold_model_paths(
        subject,
        fold,
    )
    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def load_baseline_metrics(
    path: Path,
) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline accuracy CSV not found: {path}"
        )

    metrics = {}

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            metrics[row["subject"]] = {
                "accuracy": float(row["csp_lda_accuracy"]),
            }

    load_baseline_recalls(
        metrics,
        BASELINE_RECALL_RESULTS_PATH,
    )

    missing_subjects = [
        get_subject_name(subject)
        for subject in range(1, 10)
        if get_subject_name(subject) not in metrics
        or any(
            recall_column not in metrics[get_subject_name(subject)]
            for recall_column in RECALL_COLUMNS.values()
        )
    ]

    if missing_subjects:
        raise RuntimeError(
            "Baseline metrics missing for: "
            + ", ".join(missing_subjects)
        )

    return metrics


def load_baseline_recalls(
    metrics: dict[str, dict[str, float]],
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline class-wise recall CSV not found: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["model"] == "CSP+LDA"
                and row["condition"] == "baseline"
            ):
                subject_metrics = metrics.setdefault(
                    row["subject"],
                    {},
                )
                for recall_column in RECALL_COLUMNS.values():
                    subject_metrics[recall_column] = float(
                        row[recall_column]
                    )


def save_results(
    results: list[EarlyWeightedResult],
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
                "baseline_accuracy",
                "early_weighted_accuracy",
                "accuracy_difference",
                "baseline_lh_recall",
                "early_weighted_lh_recall",
                "lh_recall_difference",
                "baseline_rh_recall",
                "early_weighted_rh_recall",
                "rh_recall_difference",
                "baseline_feet_recall",
                "early_weighted_feet_recall",
                "feet_recall_difference",
                "baseline_tongue_recall",
                "early_weighted_tongue_recall",
                "tongue_recall_difference",
                "baseline_eegnet_gap",
                "weighted_eegnet_gap",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                "subject": row.subject,
                "baseline_accuracy": f"{row.baseline_accuracy:.10g}",
                "early_weighted_accuracy": (
                    f"{row.early_weighted_accuracy:.10g}"
                ),
                "accuracy_difference": f"{row.accuracy_difference:.10g}",
                "baseline_lh_recall": _format_float(
                    row.baseline_recalls["left_hand"]
                ),
                "early_weighted_lh_recall": _format_float(
                    row.early_weighted_recalls["left_hand"]
                ),
                "lh_recall_difference": _format_float(
                    row.recall_differences["left_hand"]
                ),
                "baseline_rh_recall": f"{row.baseline_rh_recall:.10g}",
                "early_weighted_rh_recall": (
                    f"{row.early_weighted_rh_recall:.10g}"
                ),
                "rh_recall_difference": f"{row.rh_recall_difference:.10g}",
                "baseline_feet_recall": _format_float(
                    row.baseline_recalls["feet"]
                ),
                "early_weighted_feet_recall": _format_float(
                    row.early_weighted_recalls["feet"]
                ),
                "feet_recall_difference": _format_float(
                    row.recall_differences["feet"]
                ),
                "baseline_tongue_recall": _format_float(
                    row.baseline_recalls["tongue"]
                ),
                "early_weighted_tongue_recall": _format_float(
                    row.early_weighted_recalls["tongue"]
                ),
                "tongue_recall_difference": _format_float(
                    row.recall_differences["tongue"]
                ),
                "baseline_eegnet_gap": f"{row.baseline_eegnet_gap:.10g}",
                "weighted_eegnet_gap": f"{row.weighted_eegnet_gap:.10g}",
            })


def print_summary(
    results: list[EarlyWeightedResult],
) -> None:
    mean_baseline = mean(
        [row.baseline_accuracy for row in results]
    )
    mean_weighted = mean(
        [row.early_weighted_accuracy for row in results]
    )
    mean_difference = mean_weighted - mean_baseline
    mean_baseline_rh_recall = mean(
        [row.baseline_recalls["right_hand"] for row in results]
    )
    mean_weighted_rh_recall = mean(
        [row.early_weighted_recalls["right_hand"] for row in results]
    )
    mean_rh_recall_difference = (
        mean_weighted_rh_recall
        - mean_baseline_rh_recall
    )
    eegnet_gap = EEGNET_MEAN_ACCURACY - mean_weighted
    original_gap = EEGNET_MEAN_ACCURACY - REFERENCE_BASELINE_MEAN
    reduced_gap = mean_difference
    reduced_gap_fraction = reduced_gap / original_gap

    print()
    print(
        "Subject | Baseline LH | Weighted LH | Diff | "
        "Baseline RH | Weighted RH | Diff | "
        "Baseline Feet | Weighted Feet | Diff | "
        "Baseline Tongue | Weighted Tongue | Diff"
    )
    print("-" * 147)

    for row in results:
        print(
            f"{row.subject:<7} | "
            f"{row.baseline_recalls['left_hand']:>11.4f} | "
            f"{row.early_weighted_recalls['left_hand']:>11.4f} | "
            f"{row.recall_differences['left_hand']:>+7.4f} | "
            f"{row.baseline_recalls['right_hand']:>11.4f} | "
            f"{row.early_weighted_recalls['right_hand']:>11.4f} | "
            f"{row.recall_differences['right_hand']:>+7.4f} | "
            f"{row.baseline_recalls['feet']:>13.4f} | "
            f"{row.early_weighted_recalls['feet']:>13.4f} | "
            f"{row.recall_differences['feet']:>+7.4f} | "
            f"{row.baseline_recalls['tongue']:>15.4f} | "
            f"{row.early_weighted_recalls['tongue']:>15.4f} | "
            f"{row.recall_differences['tongue']:>+7.4f}"
        )

    print()
    print(f"Mean baseline CSP+LDA accuracy = {mean_baseline:.4f}")
    print(f"Mean Early-weighted CSP+LDA accuracy = {mean_weighted:.4f}")
    print(f"Mean accuracy difference = {mean_difference:.4f}")
    print()
    for class_name in RECALL_COLUMNS:
        display_name = CLASS_DISPLAY_NAMES[class_name]
        mean_baseline_recall = mean(
            [row.baseline_recalls[class_name] for row in results]
        )
        mean_weighted_recall = mean(
            [row.early_weighted_recalls[class_name] for row in results]
        )
        print(
            f"Mean baseline {display_name} recall = "
            f"{mean_baseline_recall:.4f}"
        )
        print(
            f"Mean Early-weighted {display_name} recall = "
            f"{mean_weighted_recall:.4f}"
        )
        print(f"Difference = {mean_weighted_recall - mean_baseline_recall:+.4f}")
        print()
    print()
    print(f"EEGNet gap = {eegnet_gap:.4f}")
    print(
        "Original EEGNet-CSP gap reduced by "
        f"{reduced_gap:.4f} accuracy "
        f"({reduced_gap_fraction * 100:.2f}% of original gap)"
    )


def mean(values: list[float]) -> float:
    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def class_recall(
    y_true: np.ndarray,
    predictions: np.ndarray,
    class_label: int,
) -> float:
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    class_mask = y_true == class_label

    if not np.any(class_mask):
        return float("nan")

    return float(
        np.mean(
            predictions[class_mask]
            == class_label
        )
    )


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return ""

    return f"{value:.10g}"


if __name__ == "__main__":
    main()
