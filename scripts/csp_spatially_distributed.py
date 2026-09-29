import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import mne
import numpy as np
from sklearn.metrics import accuracy_score


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.channel_entropy_statistical_analysis import _normalized_entropy
from src.analysis.channel_statistical_analysis import _relative_class_profiles
from src.analysis.csp_pattern_analysis.channel_relevance import (
    aggregate_trial_channel_relevance,
    compute_trial_channel_relevance,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_NAME_TO_LABEL
from src.data.preprocessing import BCI_2A_CHANNEL_NAMES
from src.models.csp import apply_csp
from src.models.csp_lda import predict_csp_lda, train_csp_lda
from src.utils.config import BASE_SEED, CSP_N_COMPONENTS, N_FOLDS
from src.utils.cross_validation import get_stratified_folds
from src.utils.paths import (
    SPATIALLY_SMOOTHED_CSP_RESULTS_DIR,
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_results_accuracy_comparison_path,
    get_spatially_smoothed_csp_fold_model_path,
    get_subject_name,
    get_temporal_ablation_results_path,
)


LAMBDA = 1.0
K_NEIGHBORS = 4
EEGNET_MEAN_ACCURACY = 0.7054
RESULTS_DIR_DISTRIBUTED = SPATIALLY_SMOOTHED_CSP_RESULTS_DIR / "metrics"
RESULTS_PATH = RESULTS_DIR_DISTRIBUTED / "csp_lda_distributed_results.csv"
ENTROPY_PATH = RESULTS_DIR_DISTRIBUTED / "csp_lda_distributed_entropy.csv"
BASELINE_ACCURACY_PATH = get_results_accuracy_comparison_path()
BASELINE_RECALL_PATH = get_temporal_ablation_results_path()
CLASS_ORDER = (
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
)
CLASS_DISPLAY_NAMES = {
    "left_hand": "Left-Hand",
    "right_hand": "Right-Hand",
    "feet": "Feet",
    "tongue": "Tongue",
}
RECALL_COLUMNS = {
    "left_hand": "left_hand_recall",
    "right_hand": "right_hand_recall",
    "feet": "feet_recall",
    "tongue": "tongue_recall",
}


@dataclass(frozen=True)
class DistributedResult:
    subject: str
    baseline_accuracy: float
    distributed_accuracy: float
    accuracy_difference: float
    baseline_recalls: dict[str, float]
    distributed_recalls: dict[str, float]
    recall_differences: dict[str, float]


@dataclass(frozen=True)
class EntropyResult:
    subject: str
    class_name: str
    baseline_entropy: float
    distributed_entropy: float
    entropy_difference: float


def main() -> None:
    baseline_metrics = load_baseline_metrics()
    smoothing = build_spatial_smoothing()
    print_spatial_smoothing_summary(smoothing)

    results = []
    entropy_results = []

    for subject in range(1, 10):
        subject_name = get_subject_name(subject)
        print()
        print(f"Running {subject_name}...")

        result, entropy_rows = evaluate_subject(
            subject=subject,
            baseline_metrics=baseline_metrics[subject_name],
            smoothing=smoothing,
        )
        results.append(result)
        entropy_results.extend(entropy_rows)

        print(
            f"{subject_name}: baseline={result.baseline_accuracy:.4f}, "
            f"distributed={result.distributed_accuracy:.4f}, "
            f"difference={result.accuracy_difference:+.4f}"
        )

    save_results(results, RESULTS_PATH)
    save_entropy_results(entropy_results, ENTROPY_PATH)
    print_accuracy_summary(results)
    print_class_recall_summary(results)
    print_entropy_summary(entropy_results)
    print()
    print("Saved models: models/experiments/spatially_smoothed_csp")
    print(f"Saved accuracy/recall results: {RESULTS_PATH}")
    print(f"Saved entropy results: {ENTROPY_PATH}")


def evaluate_subject(
    subject: int,
    baseline_metrics: dict[str, float],
    smoothing: dict[str, np.ndarray],
) -> tuple[DistributedResult, list[EntropyResult]]:
    data = get_data_for_subject(subject)

    if data is None:
        raise RuntimeError(
            f"Could not load data for {get_subject_name(subject)}."
        )

    X_train, y_train, X_eval, y_eval = data
    smoothing_matrix = smoothing["matrix"]

    print_sanity_checks(
        X_train,
        apply_spatial_smoothing(X_train, smoothing_matrix),
        smoothing,
    )

    models = train_or_load_distributed_csp_lda(
        subject=subject,
        X_train=X_train,
        y_train=y_train,
        smoothing_matrix=smoothing_matrix,
    )
    X_eval_distributed = apply_spatial_smoothing(
        X_eval,
        smoothing_matrix,
    )
    print_csp_feature_count(models, X_eval_distributed)

    predictions = predict_csp_lda(models, X_eval_distributed)
    distributed_accuracy = float(
        accuracy_score(y_eval, predictions)
    )
    baseline_accuracy = baseline_metrics["accuracy"]
    baseline_recalls = {
        class_name: baseline_metrics[RECALL_COLUMNS[class_name]]
        for class_name in CLASS_ORDER
    }
    distributed_recalls = compute_class_recalls(
        y_true=y_eval,
        predictions=predictions,
    )
    recall_differences = {
        class_name: (
            distributed_recalls[class_name]
            - baseline_recalls[class_name]
        )
        for class_name in CLASS_ORDER
    }

    entropy_rows = compute_entropy_results(
        subject=subject,
        distributed_models=models,
        X_eval=X_eval,
        X_eval_distributed=X_eval_distributed,
        y_eval=y_eval,
    )

    result = DistributedResult(
        subject=get_subject_name(subject),
        baseline_accuracy=baseline_accuracy,
        distributed_accuracy=distributed_accuracy,
        accuracy_difference=distributed_accuracy - baseline_accuracy,
        baseline_recalls=baseline_recalls,
        distributed_recalls=distributed_recalls,
        recall_differences=recall_differences,
    )

    return result, entropy_rows


def build_spatial_smoothing() -> dict[str, np.ndarray]:
    positions = get_channel_positions()
    adjacency = build_knn_adjacency(
        positions=positions,
        k=K_NEIGHBORS,
    )
    laplacian = normalized_graph_laplacian(adjacency)
    identity = np.eye(
        len(BCI_2A_CHANNEL_NAMES),
        dtype=np.float64,
    )
    smoothing_matrix = np.linalg.inv(
        identity
        + LAMBDA * laplacian
    )

    return {
        "positions": positions,
        "adjacency": adjacency,
        "laplacian": laplacian,
        "matrix": smoothing_matrix,
    }


def get_channel_positions() -> np.ndarray:
    montage = mne.channels.make_standard_montage(
        "standard_1020"
    )
    channel_positions = montage.get_positions()["ch_pos"]
    missing_channels = [
        channel
        for channel in BCI_2A_CHANNEL_NAMES
        if channel not in channel_positions
    ]

    if missing_channels:
        raise RuntimeError(
            "Missing standard montage positions for: "
            + ", ".join(missing_channels)
        )

    return np.asarray(
        [
            channel_positions[channel]
            for channel in BCI_2A_CHANNEL_NAMES
        ],
        dtype=np.float64,
    )


def build_knn_adjacency(
    positions: np.ndarray,
    k: int,
) -> np.ndarray:
    n_channels = positions.shape[0]
    distances = np.linalg.norm(
        positions[:, np.newaxis, :]
        - positions[np.newaxis, :, :],
        axis=2,
    )
    adjacency = np.zeros(
        (n_channels, n_channels),
        dtype=np.float64,
    )

    for channel_index in range(n_channels):
        neighbor_indices = np.argsort(
            distances[channel_index]
        )[1 : k + 1]
        adjacency[channel_index, neighbor_indices] = 1.0

    adjacency = np.maximum(
        adjacency,
        adjacency.T,
    )

    return adjacency


def normalized_graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degrees = np.sum(
        adjacency,
        axis=1,
    )

    if np.any(degrees <= 0):
        raise RuntimeError(
            "The spatial adjacency graph contains isolated channels."
        )

    inverse_sqrt_degrees = np.diag(
        1.0
        / np.sqrt(degrees)
    )
    identity = np.eye(
        adjacency.shape[0],
        dtype=np.float64,
    )

    return (
        identity
        - inverse_sqrt_degrees
        @ adjacency
        @ inverse_sqrt_degrees
    )


def apply_spatial_smoothing(
    X: np.ndarray,
    smoothing_matrix: np.ndarray,
) -> np.ndarray:
    smoothed = np.einsum(
        "cd,tdn->tcn",
        smoothing_matrix,
        X,
    )

    if smoothed.shape != X.shape:
        raise RuntimeError(
            f"Spatial smoothing changed shape from {X.shape} "
            f"to {smoothed.shape}."
        )

    return smoothed


def train_or_load_distributed_csp_lda(
    subject: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    smoothing_matrix: np.ndarray,
) -> list:
    seed = BASE_SEED + subject
    models = []

    for fold, train_idx, _ in get_stratified_folds(
        X_train,
        y_train,
        seed,
    ):
        saved_models = load_fold_models(subject, fold)

        if saved_models is not None:
            print(
                f"[LOAD] {get_subject_name(subject)} "
                f"Distributed CSP+LDA fold {fold}/{N_FOLDS}"
            )
            models.append(saved_models)
            continue

        print(
            f"[TRAIN] {get_subject_name(subject)} "
            f"Distributed CSP+LDA fold {fold}/{N_FOLDS}"
        )

        X_tr = apply_spatial_smoothing(
            X_train[train_idx],
            smoothing_matrix,
        )
        y_tr = y_train[train_idx]
        csp, lda = train_csp_lda(
            X_tr,
            y_tr,
            n_components=CSP_N_COMPONENTS,
        )
        check_csp_feature_count(csp, X_tr)
        save_fold_models(subject, fold, csp, lda)
        models.append((csp, lda))

    return models


def compute_entropy_results(
    subject: int,
    distributed_models: list,
    X_eval: np.ndarray,
    X_eval_distributed: np.ndarray,
    y_eval: np.ndarray,
) -> list[EntropyResult]:
    baseline_models = load_baseline_models(subject)
    baseline_profiles = compute_normalized_channel_profiles(
        models=baseline_models,
        X_eval=X_eval,
        y_eval=y_eval,
    )
    distributed_profiles = compute_normalized_channel_profiles(
        models=distributed_models,
        X_eval=X_eval_distributed,
        y_eval=y_eval,
    )
    baseline_entropies = compute_class_entropies(
        baseline_profiles
    )
    distributed_entropies = compute_class_entropies(
        distributed_profiles
    )

    return [
        EntropyResult(
            subject=get_subject_name(subject),
            class_name=class_name,
            baseline_entropy=baseline_entropies[class_index],
            distributed_entropy=distributed_entropies[class_index],
            entropy_difference=(
                distributed_entropies[class_index]
                - baseline_entropies[class_index]
            ),
        )
        for class_index, class_name in enumerate(CLASS_ORDER)
    ]


def compute_normalized_channel_profiles(
    models: list,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> np.ndarray:
    csps = [
        csp
        for csp, _ in models
    ]
    ldas = [
        lda
        for _, lda in models
    ]
    relevance_result = compute_trial_channel_relevance(
        csps=csps,
        ldas=ldas,
        data=X_eval,
        labels=y_eval,
    )
    class_relevance, _ = aggregate_trial_channel_relevance(
        result=relevance_result,
        mask=np.ones(
            len(y_eval),
            dtype=bool,
        ),
    )

    return _relative_class_profiles(class_relevance)


def compute_class_entropies(
    normalized_profiles: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _normalized_entropy(class_profile)
            for class_profile in normalized_profiles
        ],
        dtype=np.float64,
    )


def load_baseline_models(subject: int) -> list:
    models = []

    for fold in range(1, N_FOLDS + 1):
        csp_path = get_csp_fold_model_path(subject, fold)
        lda_path = get_lda_fold_model_path(subject, fold)

        if not (csp_path.exists() and lda_path.exists()):
            raise FileNotFoundError(
                "Missing baseline CSP+LDA fold model for "
                f"{get_subject_name(subject)} fold {fold}."
            )

        models.append((
            joblib.load(csp_path),
            joblib.load(lda_path),
        ))

    return models


def get_fold_model_paths(
    subject: int,
    fold: int,
) -> tuple[Path, Path]:
    return (
        get_spatially_smoothed_csp_fold_model_path(
            subject,
            fold,
            "csp",
        ),
        get_spatially_smoothed_csp_fold_model_path(
            subject,
            fold,
            "lda",
        ),
    )


def load_fold_models(
    subject: int,
    fold: int,
):
    csp_path, lda_path = get_fold_model_paths(subject, fold)

    if not (csp_path.exists() and lda_path.exists()):
        return None

    return joblib.load(csp_path), joblib.load(lda_path)


def save_fold_models(
    subject: int,
    fold: int,
    csp,
    lda,
) -> None:
    csp_path, lda_path = get_fold_model_paths(subject, fold)
    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def compute_class_recalls(
    y_true: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        class_name: class_recall(
            y_true=y_true,
            predictions=predictions,
            class_label=CLASS_NAME_TO_LABEL[class_name],
        )
        for class_name in CLASS_ORDER
    }


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

    return float(np.mean(predictions[class_mask] == class_label))


def print_spatial_smoothing_summary(
    smoothing: dict[str, np.ndarray],
) -> None:
    adjacency = smoothing["adjacency"]
    smoothing_matrix = smoothing["matrix"]
    print("Spatially distributed CSP+LDA")
    print(f"channel ordering = {', '.join(BCI_2A_CHANNEL_NAMES)}")
    print(f"lambda used = {LAMBDA:.4f}")
    print(f"k-nearest neighbors = {K_NEIGHBORS}")
    print(f"adjacency matrix shape = {adjacency.shape}")
    print(f"smoothing matrix shape = {smoothing_matrix.shape}")


def print_sanity_checks(
    original: np.ndarray,
    distributed: np.ndarray,
    smoothing: dict[str, np.ndarray],
) -> None:
    print(f"original epoch shape = {original.shape}")
    print(f"distributed epoch shape = {distributed.shape}")
    print(f"channel ordering = {', '.join(BCI_2A_CHANNEL_NAMES)}")
    print(f"lambda used = {LAMBDA:.4f}")
    print(f"adjacency matrix shape = {smoothing['adjacency'].shape}")
    print(f"smoothing matrix shape = {smoothing['matrix'].shape}")
    print(f"dimensions unchanged = {original.shape == distributed.shape}")


def check_csp_feature_count(csp, X: np.ndarray) -> None:
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
        raise RuntimeError("No distributed CSP+LDA models were available.")

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


def load_baseline_metrics() -> dict[str, dict[str, float]]:
    if not BASELINE_ACCURACY_PATH.exists():
        raise FileNotFoundError(
            f"Baseline accuracy CSV not found: {BASELINE_ACCURACY_PATH}"
        )

    metrics = {}

    with BASELINE_ACCURACY_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            metrics[row["subject"]] = {
                "accuracy": float(row["csp_lda_accuracy"]),
            }

    if not BASELINE_RECALL_PATH.exists():
        raise FileNotFoundError(
            f"Baseline recall CSV not found: {BASELINE_RECALL_PATH}"
        )

    with BASELINE_RECALL_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["model"] != "CSP+LDA" or row["condition"] != "baseline":
                continue

            subject_metrics = metrics.setdefault(row["subject"], {})
            for recall_column in RECALL_COLUMNS.values():
                subject_metrics[recall_column] = float(row[recall_column])

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


def save_results(
    results: list[DistributedResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                "distributed_accuracy",
                "accuracy_difference",
                "baseline_lh_recall",
                "distributed_lh_recall",
                "lh_difference",
                "baseline_rh_recall",
                "distributed_rh_recall",
                "rh_difference",
                "baseline_feet_recall",
                "distributed_feet_recall",
                "feet_difference",
                "baseline_tongue_recall",
                "distributed_tongue_recall",
                "tongue_difference",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                "subject": row.subject,
                "baseline_accuracy": format_float(row.baseline_accuracy),
                "distributed_accuracy": format_float(row.distributed_accuracy),
                "accuracy_difference": format_float(row.accuracy_difference),
                "baseline_lh_recall": format_float(
                    row.baseline_recalls["left_hand"]
                ),
                "distributed_lh_recall": format_float(
                    row.distributed_recalls["left_hand"]
                ),
                "lh_difference": format_float(
                    row.recall_differences["left_hand"]
                ),
                "baseline_rh_recall": format_float(
                    row.baseline_recalls["right_hand"]
                ),
                "distributed_rh_recall": format_float(
                    row.distributed_recalls["right_hand"]
                ),
                "rh_difference": format_float(
                    row.recall_differences["right_hand"]
                ),
                "baseline_feet_recall": format_float(
                    row.baseline_recalls["feet"]
                ),
                "distributed_feet_recall": format_float(
                    row.distributed_recalls["feet"]
                ),
                "feet_difference": format_float(
                    row.recall_differences["feet"]
                ),
                "baseline_tongue_recall": format_float(
                    row.baseline_recalls["tongue"]
                ),
                "distributed_tongue_recall": format_float(
                    row.distributed_recalls["tongue"]
                ),
                "tongue_difference": format_float(
                    row.recall_differences["tongue"]
                ),
            })


def save_entropy_results(
    results: list[EntropyResult],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "class",
                "baseline_entropy",
                "distributed_entropy",
                "entropy_difference",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow({
                "subject": row.subject,
                "class": CLASS_DISPLAY_NAMES[row.class_name],
                "baseline_entropy": format_float(row.baseline_entropy),
                "distributed_entropy": format_float(row.distributed_entropy),
                "entropy_difference": format_float(row.entropy_difference),
            })


def print_accuracy_summary(results: list[DistributedResult]) -> None:
    mean_baseline = mean([row.baseline_accuracy for row in results])
    mean_distributed = mean([row.distributed_accuracy for row in results])
    mean_difference = mean_distributed - mean_baseline
    baseline_gap = EEGNET_MEAN_ACCURACY - mean_baseline
    distributed_gap = EEGNET_MEAN_ACCURACY - mean_distributed
    gap_reduction = baseline_gap - distributed_gap

    print()
    print("Subject | Baseline Acc | Distributed Acc | Acc Diff")
    print("-" * 58)

    for row in results:
        print(
            f"{row.subject:<7} | "
            f"{row.baseline_accuracy:>12.4f} | "
            f"{row.distributed_accuracy:>15.4f} | "
            f"{row.accuracy_difference:>+8.4f}"
        )

    print()
    print(f"Mean baseline CSP+LDA accuracy = {mean_baseline:.4f}")
    print(f"Mean distributed CSP+LDA accuracy = {mean_distributed:.4f}")
    print(f"Mean accuracy difference = {mean_difference:+.4f}")
    print()
    print(f"baseline EEGNet-CSP gap = {baseline_gap:.4f}")
    print(f"distributed EEGNet-CSP gap = {distributed_gap:.4f}")
    print(f"reduction/increase in the gap = {gap_reduction:+.4f}")


def print_class_recall_summary(results: list[DistributedResult]) -> None:
    print()
    print(
        "Subject | Baseline LH | Distributed LH | Diff | "
        "Baseline RH | Distributed RH | Diff | "
        "Baseline Feet | Distributed Feet | Diff | "
        "Baseline Tongue | Distributed Tongue | Diff"
    )
    print("-" * 153)

    for row in results:
        print(
            f"{row.subject:<7} | "
            f"{row.baseline_recalls['left_hand']:>11.4f} | "
            f"{row.distributed_recalls['left_hand']:>14.4f} | "
            f"{row.recall_differences['left_hand']:>+7.4f} | "
            f"{row.baseline_recalls['right_hand']:>11.4f} | "
            f"{row.distributed_recalls['right_hand']:>14.4f} | "
            f"{row.recall_differences['right_hand']:>+7.4f} | "
            f"{row.baseline_recalls['feet']:>13.4f} | "
            f"{row.distributed_recalls['feet']:>16.4f} | "
            f"{row.recall_differences['feet']:>+7.4f} | "
            f"{row.baseline_recalls['tongue']:>15.4f} | "
            f"{row.distributed_recalls['tongue']:>18.4f} | "
            f"{row.recall_differences['tongue']:>+7.4f}"
        )

    print()

    for class_name in CLASS_ORDER:
        display_name = CLASS_DISPLAY_NAMES[class_name]
        mean_baseline_recall = mean(
            [row.baseline_recalls[class_name] for row in results]
        )
        mean_distributed_recall = mean(
            [row.distributed_recalls[class_name] for row in results]
        )
        print(
            f"Mean baseline {display_name} recall = "
            f"{mean_baseline_recall:.4f}"
        )
        print(
            f"Mean distributed {display_name} recall = "
            f"{mean_distributed_recall:.4f}"
        )
        print(f"Difference = {mean_distributed_recall - mean_baseline_recall:+.4f}")
        print()


def print_entropy_summary(results: list[EntropyResult]) -> None:
    print()
    print("Channel-Relevance Entropy")
    print("Class | Baseline mean | Distributed mean | Difference")
    print("-" * 60)

    differences = []

    for class_name in CLASS_ORDER:
        class_rows = [
            row
            for row in results
            if row.class_name == class_name
        ]
        baseline_mean = mean([
            row.baseline_entropy
            for row in class_rows
        ])
        distributed_mean = mean([
            row.distributed_entropy
            for row in class_rows
        ])
        difference = distributed_mean - baseline_mean
        differences.append(difference)
        print(
            f"{CLASS_DISPLAY_NAMES[class_name]:<10} | "
            f"{baseline_mean:>13.4f} | "
            f"{distributed_mean:>16.4f} | "
            f"{difference:>+10.4f}"
        )

    mean_entropy_difference = mean(differences)
    print()
    print(
        "Mean entropy difference across classes = "
        f"{mean_entropy_difference:+.4f}"
    )

    if mean_entropy_difference > 0:
        print("Distributed CSP produced higher mean spatial entropy.")
    else:
        print("Distributed CSP did not produce higher mean spatial entropy.")


def mean(values: list[float]) -> float:
    return float(
        np.nanmean(
            np.asarray(values, dtype=np.float64)
        )
    )


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return ""

    return f"{value:.10g}"


if __name__ == "__main__":
    main()
