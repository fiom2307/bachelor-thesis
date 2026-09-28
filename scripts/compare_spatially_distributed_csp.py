import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.analysis.csp_pattern_analysis.channel_relevance import (
    aggregate_trial_channel_relevance,
    compute_trial_channel_relevance,
)
from src.data.dataset import get_data_for_subject
from src.data.labels import CLASS_NAMES
from src.data.preprocessing import BCI_2A_CHANNEL_NAMES
from src.utils.config import BASE_SEED, N_FOLDS
from src.utils.paths import (
    MODEL_DIR,
    RESULTS_DIR,
    get_csp_fold_model_path,
    get_lda_fold_model_path,
    get_subject_name,
)
from src.visualization.common import save_figure


warnings.filterwarnings(
    "ignore",
    message="Channel names are not unique.*",
    category=RuntimeWarning,
)

MODEL_BASELINE = "Baseline CSP+LDA"
MODEL_DISTRIBUTED = "Distributed CSP+LDA"
ANALYSIS_COMMON = "common_correct"
ANALYSIS_OWN = "own_correct"
LAMBDA = 1.0
K_NEIGHBORS = 4
SUBJECTS = range(1, 10)
CLASS_ORDER = tuple(CLASS_NAMES)

SOURCE_PERFORMANCE_PATH = (
    RESULTS_DIR
    / "spatial_distributed"
    / "csp_lda_distributed_results.csv"
)
DISTRIBUTED_MODEL_DIR = MODEL_DIR / "csp_lda_distributed"
RESULTS_OUTPUT_DIR = RESULTS_DIR / "experiments" / "spatially_distributed_csp"
FIGURE_OUTPUT_DIR = ROOT_DIR / "figures" / "experiments" / "spatially_distributed_csp"

PERFORMANCE_SUMMARY_PATH = RESULTS_OUTPUT_DIR / "performance_summary.csv"
PER_SUBJECT_PERFORMANCE_PATH = RESULTS_OUTPUT_DIR / "per_subject_performance.csv"
CLASSWISE_ENTROPY_PATH = RESULTS_OUTPUT_DIR / "classwise_entropy.csv"
OVERALL_ENTROPY_PATH = RESULTS_OUTPUT_DIR / "overall_entropy.csv"
ENTROPY_STATISTICS_PATH = RESULTS_OUTPUT_DIR / "entropy_statistics.csv"
TRIAL_COUNTS_PATH = RESULTS_OUTPUT_DIR / "entropy_trial_counts.csv"
CHANNEL_RELEVANCE_PATH = RESULTS_OUTPUT_DIR / "channel_relevance_profiles.csv"

CLASSWISE_ENTROPY_FIGURE_PATH = (
    FIGURE_OUTPUT_DIR / "baseline_vs_distributed_channel_entropy.png"
)
OVERALL_ENTROPY_FIGURE_PATH = (
    FIGURE_OUTPUT_DIR / "baseline_vs_distributed_overall_channel_entropy.png"
)
CHANNEL_RELEVANCE_FIGURE_PATH = (
    FIGURE_OUTPUT_DIR / "baseline_vs_distributed_channel_relevance.png"
)


@dataclass(frozen=True)
class SubjectEntropyRow:
    subject: int
    analysis: str
    class_name: str
    baseline_entropy: float
    distributed_entropy: float
    entropy_difference: float
    baseline_n_trials: int
    distributed_n_trials: int


@dataclass(frozen=True)
class OverallEntropyRow:
    subject: int
    analysis: str
    baseline_entropy: float
    distributed_entropy: float
    entropy_difference: float


@dataclass(frozen=True)
class EntropyStatisticRow:
    analysis: str
    scope: str
    class_name: str
    n: int
    baseline_mean: float
    baseline_std: float
    distributed_mean: float
    distributed_std: float
    mean_difference: float
    wilcoxon_statistic: float
    p_value: float
    p_value_fdr: float
    significant_fdr: bool
    direction: str


@dataclass(frozen=True)
class SubjectChannelProfileRow:
    subject: int
    analysis: str
    class_name: str
    model: str
    values: np.ndarray


def main() -> None:
    performance_rows = load_performance_rows()
    performance_summary = summarize_performance(performance_rows)

    entropy_rows, overall_rows, channel_rows = collect_entropy_rows()
    statistic_rows = compute_entropy_statistics(
        entropy_rows=entropy_rows,
        overall_rows=overall_rows,
    )

    save_performance_summary(performance_summary, PERFORMANCE_SUMMARY_PATH)
    save_per_subject_performance(performance_rows, PER_SUBJECT_PERFORMANCE_PATH)
    save_classwise_entropy(entropy_rows, CLASSWISE_ENTROPY_PATH)
    save_overall_entropy(overall_rows, OVERALL_ENTROPY_PATH)
    save_entropy_statistics(statistic_rows, ENTROPY_STATISTICS_PATH)
    save_trial_counts(entropy_rows, TRIAL_COUNTS_PATH)
    save_channel_profiles(channel_rows, CHANNEL_RELEVANCE_PATH)

    plot_classwise_entropy(
        entropy_rows=entropy_rows,
        statistic_rows=statistic_rows,
        analysis=ANALYSIS_COMMON,
        output_path=CLASSWISE_ENTROPY_FIGURE_PATH,
    )
    plot_overall_entropy(
        overall_rows=overall_rows,
        statistic_rows=statistic_rows,
        analysis=ANALYSIS_COMMON,
        output_path=OVERALL_ENTROPY_FIGURE_PATH,
    )
    plot_channel_relevance(
        channel_rows=channel_rows,
        analysis=ANALYSIS_COMMON,
        output_path=CHANNEL_RELEVANCE_FIGURE_PATH,
    )

    print_final_summary(
        performance_summary=performance_summary,
        statistic_rows=statistic_rows,
    )


def collect_entropy_rows() -> tuple[
    list[SubjectEntropyRow],
    list[OverallEntropyRow],
    list[SubjectChannelProfileRow],
]:
    smoothing_matrix = load_distributed_smoothing_matrix()
    entropy_rows = []
    overall_rows = []
    channel_rows = []

    for subject in SUBJECTS:
        data = get_data_for_subject(subject)

        if data is None:
            raise RuntimeError(
                f"Could not load data for {get_subject_name(subject)}."
            )

        _, _, x_eval, y_eval = data
        x_eval_distributed = apply_spatial_smoothing(x_eval, smoothing_matrix)

        baseline_models = load_baseline_models(subject)
        distributed_models = load_distributed_models(subject)

        baseline_result = compute_relevance_result(baseline_models, x_eval, y_eval)
        distributed_result = compute_relevance_result(
            distributed_models,
            x_eval_distributed,
            y_eval,
        )

        selections = (
            (
                ANALYSIS_COMMON,
                baseline_result.correct_mask & distributed_result.correct_mask,
                baseline_result.correct_mask & distributed_result.correct_mask,
            ),
            (
                ANALYSIS_OWN,
                baseline_result.correct_mask,
                distributed_result.correct_mask,
            ),
        )

        for analysis, baseline_mask, distributed_mask in selections:
            baseline_profiles, baseline_counts = class_profiles(
                baseline_result,
                baseline_mask,
            )
            distributed_profiles, distributed_counts = class_profiles(
                distributed_result,
                distributed_mask,
            )
            baseline_entropies = class_entropies(baseline_profiles)
            distributed_entropies = class_entropies(distributed_profiles)

            entropy_rows.extend(
                SubjectEntropyRow(
                    subject=subject,
                    analysis=analysis,
                    class_name=class_name,
                    baseline_entropy=baseline_entropies[class_index],
                    distributed_entropy=distributed_entropies[class_index],
                    entropy_difference=(
                        distributed_entropies[class_index]
                        - baseline_entropies[class_index]
                    ),
                    baseline_n_trials=int(baseline_counts[class_index]),
                    distributed_n_trials=int(distributed_counts[class_index]),
                )
                for class_index, class_name in enumerate(CLASS_ORDER)
            )

            overall_rows.append(
                OverallEntropyRow(
                    subject=subject,
                    analysis=analysis,
                    baseline_entropy=class_balanced_entropy(baseline_entropies),
                    distributed_entropy=class_balanced_entropy(
                        distributed_entropies
                    ),
                    entropy_difference=(
                        class_balanced_entropy(distributed_entropies)
                        - class_balanced_entropy(baseline_entropies)
                    ),
                )
            )

            channel_rows.extend(
                make_channel_rows(
                    subject=subject,
                    analysis=analysis,
                    model=MODEL_BASELINE,
                    profiles=baseline_profiles,
                )
            )
            channel_rows.extend(
                make_channel_rows(
                    subject=subject,
                    analysis=analysis,
                    model=MODEL_DISTRIBUTED,
                    profiles=distributed_profiles,
                )
            )

    return entropy_rows, overall_rows, channel_rows


def load_distributed_smoothing_matrix() -> np.ndarray:
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

    return np.linalg.inv(identity + LAMBDA * laplacian)


def get_channel_positions() -> np.ndarray:
    montage = mne.channels.make_standard_montage("standard_1020")
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


def build_knn_adjacency(positions: np.ndarray, k: int) -> np.ndarray:
    n_channels = positions.shape[0]
    distances = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :],
        axis=2,
    )
    adjacency = np.zeros(
        (n_channels, n_channels),
        dtype=np.float64,
    )

    for channel_index in range(n_channels):
        neighbor_indices = np.argsort(distances[channel_index])[1 : k + 1]
        adjacency[channel_index, neighbor_indices] = 1.0

    return np.maximum(adjacency, adjacency.T)


def normalized_graph_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degrees = np.sum(adjacency, axis=1)

    if np.any(degrees <= 0):
        raise RuntimeError(
            "The spatial adjacency graph contains isolated channels."
        )

    inverse_sqrt_degrees = np.diag(1.0 / np.sqrt(degrees))
    identity = np.eye(adjacency.shape[0], dtype=np.float64)

    return identity - inverse_sqrt_degrees @ adjacency @ inverse_sqrt_degrees


def apply_spatial_smoothing(
    x: np.ndarray,
    smoothing_matrix: np.ndarray,
) -> np.ndarray:
    smoothed = np.einsum(
        "cd,tdn->tcn",
        smoothing_matrix,
        x,
    )

    if smoothed.shape != x.shape:
        raise RuntimeError(
            f"Spatial smoothing changed shape from {x.shape} to {smoothed.shape}."
        )

    return smoothed


def compute_relevance_result(models: list, x_eval: np.ndarray, y_eval: np.ndarray):
    return compute_trial_channel_relevance(
        csps=[csp for csp, _ in models],
        ldas=[lda for _, lda in models],
        data=x_eval,
        labels=y_eval,
    )


def class_profiles(result, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_relevance, counts = aggregate_trial_channel_relevance(
        result=result,
        mask=mask,
    )

    return _relative_class_profiles(class_relevance), counts


def class_entropies(profiles: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            _normalized_entropy(profiles[class_index])
            for class_index in range(len(CLASS_ORDER))
        ],
        dtype=np.float64,
    )


def class_balanced_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)

    if np.any(~np.isfinite(values)):
        return np.nan

    return float(np.mean(values))


def make_channel_rows(
    subject: int,
    analysis: str,
    model: str,
    profiles: np.ndarray,
) -> list[SubjectChannelProfileRow]:
    return [
        SubjectChannelProfileRow(
            subject=subject,
            analysis=analysis,
            class_name=class_name,
            model=model,
            values=np.asarray(profiles[class_index], dtype=np.float64),
        )
        for class_index, class_name in enumerate(CLASS_ORDER)
    ]


def load_baseline_models(subject: int) -> list:
    models = []

    for fold in range(1, N_FOLDS + 1):
        csp_path = get_csp_fold_model_path(subject, fold)
        lda_path = get_lda_fold_model_path(subject, fold)
        require_existing_file(csp_path)
        require_existing_file(lda_path)
        models.append((joblib.load(csp_path), joblib.load(lda_path)))

    return models


def load_distributed_models(subject: int) -> list:
    subject_name = get_subject_name(subject)
    models = []

    for fold in range(1, N_FOLDS + 1):
        csp_path = (
            DISTRIBUTED_MODEL_DIR
            / subject_name
            / f"{subject_name}_csp_kfold_seed{BASE_SEED}_fold{fold}.joblib"
        )
        lda_path = (
            DISTRIBUTED_MODEL_DIR
            / subject_name
            / f"{subject_name}_lda_kfold_seed{BASE_SEED}_fold{fold}.joblib"
        )
        require_existing_file(csp_path)
        require_existing_file(lda_path)
        models.append((joblib.load(csp_path), joblib.load(lda_path)))

    return models


def require_existing_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Required saved artifact is missing; refusing to retrain: "
            f"{path}"
        )


def compute_entropy_statistics(
    entropy_rows: list[SubjectEntropyRow],
    overall_rows: list[OverallEntropyRow],
) -> list[EntropyStatisticRow]:
    rows = []

    for analysis in (ANALYSIS_COMMON, ANALYSIS_OWN):
        rows.append(
            compute_overall_entropy_statistic(
                [
                    row
                    for row in overall_rows
                    if row.analysis == analysis
                ]
            )
        )

        class_rows = [
            compute_class_entropy_statistic(
                [
                    row
                    for row in entropy_rows
                    if row.analysis == analysis and row.class_name == class_name
                ],
                class_name,
            )
            for class_name in CLASS_ORDER
        ]
        corrected_p_values, significant = _fdr_correct(
            [row.p_value for row in class_rows]
        )
        rows.extend(
            replace_fdr(row, q_value, is_significant)
            for row, q_value, is_significant in zip(
                class_rows,
                corrected_p_values,
                significant,
                strict=True,
            )
        )

    return rows


def compute_overall_entropy_statistic(
    rows: list[OverallEntropyRow],
) -> EntropyStatisticRow:
    baseline_values = np.asarray(
        [row.baseline_entropy for row in rows],
        dtype=np.float64,
    )
    distributed_values = np.asarray(
        [row.distributed_entropy for row in rows],
        dtype=np.float64,
    )

    row = make_statistic_row(
        analysis=rows[0].analysis,
        scope="overall",
        class_name="all_classes_balanced",
        baseline_values=baseline_values,
        distributed_values=distributed_values,
    )

    q_values, significant = _fdr_correct([row.p_value])

    return replace_fdr(row, q_values[0], bool(significant[0]))


def compute_class_entropy_statistic(
    rows: list[SubjectEntropyRow],
    class_name: str,
) -> EntropyStatisticRow:
    return make_statistic_row(
        analysis=rows[0].analysis,
        scope="classwise",
        class_name=class_name,
        baseline_values=np.asarray(
            [row.baseline_entropy for row in rows],
            dtype=np.float64,
        ),
        distributed_values=np.asarray(
            [row.distributed_entropy for row in rows],
            dtype=np.float64,
        ),
    )


def make_statistic_row(
    analysis: str,
    scope: str,
    class_name: str,
    baseline_values: np.ndarray,
    distributed_values: np.ndarray,
) -> EntropyStatisticRow:
    finite_mask = (
        np.isfinite(baseline_values)
        & np.isfinite(distributed_values)
    )
    baseline_values = baseline_values[finite_mask]
    distributed_values = distributed_values[finite_mask]
    differences = distributed_values - baseline_values
    statistic, p_value = _paired_wilcoxon(differences)
    mean_difference = _nanmean(differences)

    return EntropyStatisticRow(
        analysis=analysis,
        scope=scope,
        class_name=class_name,
        n=len(differences),
        baseline_mean=_nanmean(baseline_values),
        baseline_std=_nanstd(baseline_values),
        distributed_mean=_nanmean(distributed_values),
        distributed_std=_nanstd(distributed_values),
        mean_difference=mean_difference,
        wilcoxon_statistic=statistic,
        p_value=p_value,
        p_value_fdr=np.nan,
        significant_fdr=False,
        direction=direction_label(mean_difference),
    )


def replace_fdr(
    row: EntropyStatisticRow,
    q_value: float,
    significant: bool,
) -> EntropyStatisticRow:
    return EntropyStatisticRow(
        analysis=row.analysis,
        scope=row.scope,
        class_name=row.class_name,
        n=row.n,
        baseline_mean=row.baseline_mean,
        baseline_std=row.baseline_std,
        distributed_mean=row.distributed_mean,
        distributed_std=row.distributed_std,
        mean_difference=row.mean_difference,
        wilcoxon_statistic=row.wilcoxon_statistic,
        p_value=row.p_value,
        p_value_fdr=float(q_value),
        significant_fdr=bool(significant),
        direction=row.direction,
    )


def direction_label(mean_difference: float) -> str:
    if not np.isfinite(mean_difference):
        return "not available"

    if mean_difference > 0:
        return "Distributed CSP+LDA > Baseline CSP+LDA"

    if mean_difference < 0:
        return "Distributed CSP+LDA < Baseline CSP+LDA"

    return "Distributed CSP+LDA = Baseline CSP+LDA"


def load_performance_rows() -> list[dict[str, str]]:
    require_existing_file(SOURCE_PERFORMANCE_PATH)

    with SOURCE_PERFORMANCE_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_performance(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    metric_specs = (
        ("Accuracy", "baseline_accuracy", "distributed_accuracy"),
        ("LH Recall", "baseline_lh_recall", "distributed_lh_recall"),
        ("RH Recall", "baseline_rh_recall", "distributed_rh_recall"),
        ("Feet Recall", "baseline_feet_recall", "distributed_feet_recall"),
        ("Tongue Recall", "baseline_tongue_recall", "distributed_tongue_recall"),
    )

    summary = []

    for metric, baseline_column, distributed_column in metric_specs:
        baseline_values = np.asarray(
            [float(row[baseline_column]) for row in rows],
            dtype=np.float64,
        )
        distributed_values = np.asarray(
            [float(row[distributed_column]) for row in rows],
            dtype=np.float64,
        )
        baseline_mean = float(np.mean(baseline_values))
        distributed_mean = float(np.mean(distributed_values))

        summary.append(
            {
                "metric": metric,
                "baseline": baseline_mean,
                "distributed": distributed_mean,
                "delta": distributed_mean - baseline_mean,
            }
        )

    return summary


def save_performance_summary(
    rows: list[dict[str, float | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "baseline_csp_lda",
                "distributed_csp_lda",
                "delta",
                "baseline_percent",
                "distributed_percent",
                "delta_percentage_points",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "baseline_csp_lda": _format_float(row["baseline"]),
                    "distributed_csp_lda": _format_float(row["distributed"]),
                    "delta": _format_float(row["delta"]),
                    "baseline_percent": _format_float(row["baseline"] * 100.0),
                    "distributed_percent": _format_float(
                        row["distributed"] * 100.0
                    ),
                    "delta_percentage_points": _format_float(row["delta"] * 100.0),
                }
            )


def save_per_subject_performance(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_classwise_entropy(
    rows: list[SubjectEntropyRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Subject",
                "Analysis",
                "Class",
                "Baseline_Entropy",
                "Distributed_Entropy",
                "Entropy_Difference",
                "Baseline_N_Trials",
                "Distributed_N_Trials",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "Subject": get_subject_name(row.subject),
                    "Analysis": row.analysis,
                    "Class": row.class_name,
                    "Baseline_Entropy": _format_float(row.baseline_entropy),
                    "Distributed_Entropy": _format_float(row.distributed_entropy),
                    "Entropy_Difference": _format_float(row.entropy_difference),
                    "Baseline_N_Trials": row.baseline_n_trials,
                    "Distributed_N_Trials": row.distributed_n_trials,
                }
            )


def save_overall_entropy(
    rows: list[OverallEntropyRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Subject",
                "Analysis",
                "Baseline_Entropy",
                "Distributed_Entropy",
                "Entropy_Difference",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "Subject": get_subject_name(row.subject),
                    "Analysis": row.analysis,
                    "Baseline_Entropy": _format_float(row.baseline_entropy),
                    "Distributed_Entropy": _format_float(row.distributed_entropy),
                    "Entropy_Difference": _format_float(row.entropy_difference),
                }
            )


def save_entropy_statistics(
    rows: list[EntropyStatisticRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "analysis",
                "scope",
                "class",
                "n",
                "baseline_mean",
                "baseline_std",
                "distributed_mean",
                "distributed_std",
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
            writer.writerow(
                {
                    "analysis": row.analysis,
                    "scope": row.scope,
                    "class": row.class_name,
                    "n": row.n,
                    "baseline_mean": _format_float(row.baseline_mean),
                    "baseline_std": _format_float(row.baseline_std),
                    "distributed_mean": _format_float(row.distributed_mean),
                    "distributed_std": _format_float(row.distributed_std),
                    "mean_difference": _format_float(row.mean_difference),
                    "wilcoxon_statistic": _format_float(row.wilcoxon_statistic),
                    "p_value": _format_float(row.p_value),
                    "p_value_fdr": _format_float(row.p_value_fdr),
                    "significant_fdr": row.significant_fdr,
                    "direction": row.direction,
                }
            )


def save_trial_counts(
    rows: list[SubjectEntropyRow],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "analysis",
                "class",
                "baseline_n_trials",
                "distributed_n_trials",
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "subject": get_subject_name(row.subject),
                    "analysis": row.analysis,
                    "class": row.class_name,
                    "baseline_n_trials": row.baseline_n_trials,
                    "distributed_n_trials": row.distributed_n_trials,
                }
            )


def save_channel_profiles(
    rows: list[SubjectChannelProfileRow],
    output_path: Path,
) -> None:
    from src.data.preprocessing import BCI_2A_CHANNEL_NAMES

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "analysis",
                "class",
                "model",
                *BCI_2A_CHANNEL_NAMES,
            ],
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "subject": get_subject_name(row.subject),
                    "analysis": row.analysis,
                    "class": row.class_name,
                    "model": row.model,
                    **{
                        channel: _format_float(value)
                        for channel, value in zip(
                            BCI_2A_CHANNEL_NAMES,
                            row.values,
                            strict=True,
                        )
                    },
                }
            )


def plot_classwise_entropy(
    entropy_rows: list[SubjectEntropyRow],
    statistic_rows: list[EntropyStatisticRow],
    analysis: str,
    output_path: Path,
) -> None:
    class_rows = [
        row
        for row in statistic_rows
        if row.analysis == analysis and row.scope == "classwise"
    ]
    x_positions = np.arange(len(CLASS_ORDER))
    offset = 0.12
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)

    baseline_values = classwise_matrix(
        entropy_rows,
        analysis,
        "baseline_entropy",
    )
    distributed_values = classwise_matrix(
        entropy_rows,
        analysis,
        "distributed_entropy",
    )

    axis.errorbar(
        x_positions - offset,
        np.nanmean(baseline_values, axis=0),
        yerr=np.nanstd(baseline_values, axis=0, ddof=1),
        marker="o",
        capsize=3,
        linewidth=2,
        label=MODEL_BASELINE,
    )
    axis.errorbar(
        x_positions + offset,
        np.nanmean(distributed_values, axis=0),
        yerr=np.nanstd(distributed_values, axis=0, ddof=1),
        marker="o",
        capsize=3,
        linewidth=2,
        label=MODEL_DISTRIBUTED,
    )

    axis.set_ylim(0.0, 1.05)

    for class_index, class_name in enumerate(CLASS_ORDER):
        row = next(row for row in class_rows if row.class_name == class_name)
        axis.text(
            class_index,
            0.98,
            significance_label(row),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(CLASS_ORDER, rotation=20, ha="right")
    axis.set_ylabel("Normalized channel entropy")
    axis.set_title("Baseline vs distributed CSP+LDA channel entropy")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)
    plt.close(figure)


def plot_overall_entropy(
    overall_rows: list[OverallEntropyRow],
    statistic_rows: list[EntropyStatisticRow],
    analysis: str,
    output_path: Path,
) -> None:
    selected = [
        row
        for row in overall_rows
        if row.analysis == analysis
    ]
    statistic = next(
        row
        for row in statistic_rows
        if row.analysis == analysis and row.scope == "overall"
    )
    baseline_values = np.asarray(
        [row.baseline_entropy for row in selected],
        dtype=np.float64,
    )
    distributed_values = np.asarray(
        [row.distributed_entropy for row in selected],
        dtype=np.float64,
    )
    x_positions = np.asarray([0.0])
    offset = 0.12
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)

    axis.errorbar(
        x_positions - offset,
        [np.nanmean(baseline_values)],
        yerr=[np.nanstd(baseline_values, ddof=1)],
        marker="o",
        capsize=3,
        linewidth=2,
        label=MODEL_BASELINE,
    )
    axis.errorbar(
        x_positions + offset,
        [np.nanmean(distributed_values)],
        yerr=[np.nanstd(distributed_values, ddof=1)],
        marker="o",
        capsize=3,
        linewidth=2,
        label=MODEL_DISTRIBUTED,
    )

    axis.text(
        0,
        0.98,
        significance_label(statistic),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks([0])
    axis.set_xticklabels(["Overall"])
    axis.set_ylabel("Normalized channel entropy")
    axis.set_title("Overall channel entropy")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(figure, output_path)
    plt.close(figure)


def plot_channel_relevance(
    channel_rows: list[SubjectChannelProfileRow],
    analysis: str,
    output_path: Path,
) -> None:
    from src.data.preprocessing import BCI_2A_CHANNEL_NAMES

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(14, 5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for col, class_name in enumerate(CLASS_ORDER):
        for row_index, model in enumerate((MODEL_BASELINE, MODEL_DISTRIBUTED)):
            axis = axes[row_index, col]
            values = np.stack(
                [
                    row.values
                    for row in channel_rows
                    if (
                        row.analysis == analysis
                        and row.class_name == class_name
                        and row.model == model
                    )
                ]
            )
            mean_values = np.nanmean(values, axis=0)
            axis.bar(np.arange(len(BCI_2A_CHANNEL_NAMES)), mean_values)
            axis.set_title(
                f"{class_name}\n{model}"
                if row_index == 0
                else model
            )
            axis.grid(axis="y", alpha=0.25)

    for axis in axes[-1, :]:
        axis.set_xticks(np.arange(len(BCI_2A_CHANNEL_NAMES)))
        axis.set_xticklabels(
            BCI_2A_CHANNEL_NAMES,
            rotation=90,
            fontsize=7,
        )

    for axis in axes[:, 0]:
        axis.set_ylabel("Mean normalized relevance")

    save_figure(figure, output_path)
    plt.close(figure)


def classwise_matrix(
    rows: list[SubjectEntropyRow],
    analysis: str,
    attribute: str,
) -> np.ndarray:
    matrix = np.full(
        (
            len(tuple(SUBJECTS)),
            len(CLASS_ORDER),
        ),
        np.nan,
        dtype=np.float64,
    )

    for row in rows:
        if row.analysis != analysis:
            continue

        subject_index = row.subject - 1
        class_index = CLASS_ORDER.index(row.class_name)
        matrix[subject_index, class_index] = getattr(row, attribute)

    return matrix


def significance_label(row: EntropyStatisticRow) -> str:
    if not np.isfinite(row.p_value_fdr):
        return ""

    if row.p_value_fdr < 0.001:
        return f"***\nq={row.p_value_fdr:.2g}"

    if row.p_value_fdr < 0.01:
        return f"**\nq={row.p_value_fdr:.2g}"

    if row.p_value_fdr < 0.05:
        return f"*\nq={row.p_value_fdr:.2g}"

    return f"q={row.p_value_fdr:.2g}"


def _relative_class_profiles(class_relevance: np.ndarray) -> np.ndarray:
    class_relevance = np.asarray(
        class_relevance,
        dtype=np.float64,
    )
    normalized = np.full_like(
        class_relevance,
        np.nan,
        dtype=np.float64,
    )

    for class_index in range(class_relevance.shape[0]):
        values = class_relevance[class_index]
        denominator = np.nansum(values)

        if not np.isfinite(denominator) or denominator <= 0:
            continue

        normalized[class_index] = values / denominator

    return normalized


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or len(values) < 2
        or np.any(~np.isfinite(values))
    ):
        return np.nan

    positive_values = values[values > 0]

    if len(positive_values) == 0:
        return np.nan

    entropy = -np.sum(positive_values * np.log(positive_values))

    return float(entropy / np.log(len(values)))


def _paired_wilcoxon(differences: np.ndarray) -> tuple[float, float]:
    differences = np.asarray(
        differences,
        dtype=np.float64,
    )
    differences = differences[np.isfinite(differences)]

    if len(differences) < 2:
        return np.nan, np.nan

    if np.allclose(differences, 0.0):
        return 0.0, 1.0

    statistic, p_value = wilcoxon(
        differences,
        alternative="two-sided",
    )

    return float(statistic), float(p_value)


def _fdr_correct(p_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
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
    finite_mask = np.isfinite(p_values_array)

    if np.any(finite_mask):
        reject, corrected_values, _, _ = multipletests(
            p_values_array[finite_mask],
            alpha=0.05,
            method="fdr_bh",
        )
        corrected[finite_mask] = corrected_values
        significant[finite_mask] = reject

    return corrected, significant


def _nanmean(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan

    return float(np.nanmean(values))


def _nanstd(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan

    return float(np.nanstd(values, ddof=1))


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return ""

    return f"{value:.10g}"


def print_final_summary(
    performance_summary: list[dict[str, float | str]],
    statistic_rows: list[EntropyStatisticRow],
) -> None:
    accuracy = next(row for row in performance_summary if row["metric"] == "Accuracy")

    print("Baseline vs distributed CSP mean accuracy")
    print(
        f"{percent(accuracy['baseline'])} vs "
        f"{percent(accuracy['distributed'])}"
    )
    print("Accuracy change in percentage points")
    print(f"{percentage_points(accuracy['delta'])}")
    print("Class-wise recall changes")

    for metric in ("LH Recall", "RH Recall", "Feet Recall", "Tongue Recall"):
        row = next(item for item in performance_summary if item["metric"] == metric)
        print(f"{metric}: {percentage_points(row['delta'])}")

    common_rows = [
        row
        for row in statistic_rows
        if row.analysis == ANALYSIS_COMMON
    ]
    overall = next(row for row in common_rows if row.scope == "overall")

    print("Overall baseline vs distributed entropy")
    print(
        f"{overall.baseline_mean:.4f} vs "
        f"{overall.distributed_mean:.4f}; "
        f"difference {overall.mean_difference:+.4f}"
    )
    print("Overall entropy statistical result")
    print(
        f"p={overall.p_value:.4g}, "
        f"q={overall.p_value_fdr:.4g}, "
        f"{overall.direction}"
    )
    print("Class-wise baseline vs distributed entropy")

    for row in common_rows:
        if row.scope != "classwise":
            continue
        print(
            f"{row.class_name}: {row.baseline_mean:.4f} vs "
            f"{row.distributed_mean:.4f}; "
            f"difference {row.mean_difference:+.4f}"
        )

    print("Class-wise p-values and FDR-corrected q-values")

    for row in common_rows:
        if row.scope != "classwise":
            continue
        print(
            f"{row.class_name}: p={row.p_value:.4g}, "
            f"q={row.p_value_fdr:.4g}"
        )

    print("Whether distributed CSP produced higher entropy overall and/or class-wise")
    print(f"Overall: {'higher' if overall.mean_difference > 0 else 'lower or equal'}")

    for row in common_rows:
        if row.scope != "classwise":
            continue
        direction = "higher" if row.mean_difference > 0 else "lower or equal"
        print(f"{row.class_name}: {direction}")

    print("Paths of the generated CSV files")
    for path in (
        PERFORMANCE_SUMMARY_PATH,
        PER_SUBJECT_PERFORMANCE_PATH,
        CLASSWISE_ENTROPY_PATH,
        OVERALL_ENTROPY_PATH,
        ENTROPY_STATISTICS_PATH,
        TRIAL_COUNTS_PATH,
        CHANNEL_RELEVANCE_PATH,
    ):
        print(path)

    print("Paths of the generated figures")
    for path in (
        CLASSWISE_ENTROPY_FIGURE_PATH,
        OVERALL_ENTROPY_FIGURE_PATH,
        CHANNEL_RELEVANCE_FIGURE_PATH,
    ):
        print(path)


def percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def percentage_points(value: float) -> str:
    return f"{value * 100.0:+.1f} pp"


if __name__ == "__main__":
    main()
