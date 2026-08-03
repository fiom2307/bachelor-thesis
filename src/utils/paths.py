from pathlib import Path
from typing import Literal

from src.utils.config import BASE_SEED


# Project directories
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

EEGNET_MODEL_DIR = MODEL_DIR / "eegnet"
CSP_LDA_MODEL_DIR = MODEL_DIR / "csp_lda"

ACCURACY_RESULTS_DIR = RESULTS_DIR / "accuracies"
CONFUSION_MATRIX_RESULTS_DIR = RESULTS_DIR / "confusion_matrices"
SPECTRAL_ANALYSIS_DIR = RESULTS_DIR / "spectral_analysis"

SHAP_RESULTS_DIR = RESULTS_DIR / "shap_analysis"
CSP_RESULTS_DIR = RESULTS_DIR / "csp_lda_occlusion_analysis"
CSP_PATTERN_RESULTS_DIR = RESULTS_DIR / "csp_pattern_analysis"


RelevanceMethod = Literal[
    "shap",
    "csp",
]

TrialSelection = Literal[
    "correct",
    "incorrect",
]

RelevancePlotType = Literal[
    "channel_time",
    "temporal_relevance",
    "topographies",
    "channel_rankings",
    "channel_relevance",
]


# Create main output directories
for directory in (
    EEGNET_MODEL_DIR,
    CSP_LDA_MODEL_DIR,
    ACCURACY_RESULTS_DIR,
    CONFUSION_MATRIX_RESULTS_DIR,
    SPECTRAL_ANALYSIS_DIR,
    SHAP_RESULTS_DIR,
    CSP_RESULTS_DIR,
    CSP_PATTERN_RESULTS_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def _create_directory(
    path: Path,
) -> Path:
    """
    Create a directory, including any missing parent directories.
    """
    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _get_seed_directory(
    base_directory: Path,
) -> Path:
    """
    Return and create the output directory for the configured seed.
    """
    return _create_directory(
        base_directory / f"seed_{BASE_SEED}"
    )


def _get_subject_spectral_subdir(
    subject: int,
    subdir: str,
) -> Path:
    """
    Return and create a spectral-analysis subdirectory for one subject.
    """
    return _create_directory(
        get_subject_spectral_dir(subject)
        / subdir
    )


def _get_relevance_results_directory(
    method: RelevanceMethod,
) -> Path:
    """
    Return the main results directory for one relevance method.
    """
    if method == "shap":
        return SHAP_RESULTS_DIR

    if method == "csp":
        return CSP_RESULTS_DIR

    raise ValueError(
        f"Unsupported relevance method: {method}"
    )


def _get_relevance_result_name(
    subject: int | None,
) -> str:
    """
    Return the filename prefix for a subject-wise or global mean result.
    """
    if subject is None:
        return "mean"

    return get_subject_name(subject)


def _get_relevance_plot_path(
    method: RelevanceMethod,
    plot_type: RelevancePlotType,
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean relevance plot path.
    """
    base_directory = (
        _get_relevance_results_directory(
            method
        )
    )

    output_directory = _create_directory(
        base_directory / plot_type
    )

    result_name = _get_relevance_result_name(
        subject
    )

    plot_filename_names = {
        "channel_time": "channel_time",
        "temporal_relevance": "temporal",
        "topographies": "topographies",
        "channel_rankings": "channel_rankings",
        "channel_relevance": "channel_relevance",
    }

    plot_filename_name = (
        plot_filename_names[plot_type]
    )

    filename = (
        f"{result_name}_"
        f"{method}_"
        f"{plot_filename_name}_"
        f"{trial_selection}.png"
    )

    return output_directory / filename


def _get_relevance_values_path(
    method: RelevanceMethod,
    subject: int,
) -> Path:
    """
    Return the saved relevance values path for one subject.
    """
    subject_name = get_subject_name(
        subject
    )

    base_directory = (
        _get_relevance_results_directory(
            method
        )
    )

    output_directory = _create_directory(
        base_directory / "values"
    )

    return (
        output_directory
        / f"{subject_name}_{method}_values.npz"
    )


def get_subject_name(
    subject: int,
) -> str:
    """
    Return the formatted name of one subject.
    """
    return f"A{subject:02d}"


def get_subject_files(
    subject: int,
) -> tuple[Path, Path, Path] | None:
    """
    Return the training, evaluation, and label files for one subject.
    """
    subject_name = get_subject_name(
        subject
    )

    train_file = (
        DATA_DIR
        / f"{subject_name}T.gdf"
    )

    eval_file = (
        DATA_DIR
        / f"{subject_name}E.gdf"
    )

    mat_file = (
        DATA_DIR
        / f"{subject_name}E.mat"
    )

    subject_files = (
        train_file,
        eval_file,
        mat_file,
    )

    if not all(
        path.exists()
        for path in subject_files
    ):
        return None

    return subject_files


def is_eval_file(
    file_path: str | Path,
) -> bool:
    """
    Check whether a file belongs to an evaluation session.
    """
    return (
        Path(file_path)
        .stem
        .upper()
        .endswith("E")
    )


# ----------------------------------------------------------------------
# CSP+LDA model paths
# ----------------------------------------------------------------------

def get_csp_lda_subject_dir(
    subject: int,
) -> Path:
    """
    Return and create the CSP+LDA model directory for one subject.
    """
    return _create_directory(
        CSP_LDA_MODEL_DIR
        / f"seed_{BASE_SEED}"
        / get_subject_name(subject)
    )


def get_csp_fold_model_path(
    subject: int,
    fold: int,
) -> Path:
    """
    Return the CSP model path for one cross-validation fold.
    """
    subject_name = get_subject_name(
        subject
    )

    filename = (
        f"{subject_name}_csp_kfold_"
        f"seed{BASE_SEED}_fold{fold}.joblib"
    )

    return (
        get_csp_lda_subject_dir(subject)
        / filename
    )


def get_lda_fold_model_path(
    subject: int,
    fold: int,
) -> Path:
    """
    Return the LDA model path for one cross-validation fold.
    """
    subject_name = get_subject_name(
        subject
    )

    filename = (
        f"{subject_name}_lda_kfold_"
        f"seed{BASE_SEED}_fold{fold}.joblib"
    )

    return (
        get_csp_lda_subject_dir(subject)
        / filename
    )


# ----------------------------------------------------------------------
# EEGNet model paths
# ----------------------------------------------------------------------

def get_eegnet_subject_dir(
    subject: int,
) -> Path:
    """
    Return and create the EEGNet model directory for one subject.
    """
    return _create_directory(
        EEGNET_MODEL_DIR
        / f"seed_{BASE_SEED}"
        / get_subject_name(subject)
    )


def get_eegnet_fold_model_path(
    subject: int,
    fold: int,
) -> Path:
    """
    Return the EEGNet model path for one cross-validation fold.
    """
    subject_name = get_subject_name(
        subject
    )

    filename = (
        f"{subject_name}_eegnet_kfold_"
        f"seed{BASE_SEED}_fold{fold}.keras"
    )

    return (
        get_eegnet_subject_dir(subject)
        / filename
    )


# ----------------------------------------------------------------------
# Evaluation result paths
# ----------------------------------------------------------------------

def get_results_accuracy_comparison_path() -> Path:
    """
    Return the CSP+LDA and EEGNet accuracy comparison CSV path.
    """
    return (
        ACCURACY_RESULTS_DIR
        / (
            f"seed_{BASE_SEED}_"
            "csp_lda_vs_eegnet.csv"
        )
    )


def get_all_confusion_matrices_path() -> Path:
    """
    Return the confusion-matrix figure path for all subjects.
    """
    output_directory = _get_seed_directory(
        CONFUSION_MATRIX_RESULTS_DIR
    )

    return (
        output_directory
        / (
            f"seed_{BASE_SEED}_"
            "all_confusion_matrices.png"
        )
    )


def get_subject_confusion_matrices_path(
    subject: int,
) -> Path:
    """
    Return the confusion-matrix figure path for one subject.
    """
    subject_name = get_subject_name(
        subject
    )

    output_directory = _get_seed_directory(
        CONFUSION_MATRIX_RESULTS_DIR
    )

    return (
        output_directory
        / (
            f"seed_{BASE_SEED}_"
            f"{subject_name}_"
            "confusion_matrices.png"
        )
    )


# ----------------------------------------------------------------------
# Spectral analysis paths
# ----------------------------------------------------------------------

def get_subject_spectral_dir(
    subject: int,
) -> Path:
    """
    Return the spectral-analysis directory for one subject.
    """
    return (
        SPECTRAL_ANALYSIS_DIR
        / get_subject_name(subject)
    )


def get_erd_topographies_path(
    subject: int,
) -> Path:
    """
    Return the class-wise ERD/ERS topography figure path.
    """
    output_directory = (
        _get_subject_spectral_subdir(
            subject,
            "topographies",
        )
    )

    return (
        output_directory
        / "classwise_erd_ers.png"
    )


def get_tfr_path(
    subject: int,
    channel: str,
) -> Path:
    """
    Return the time-frequency representation figure path.
    """
    output_directory = (
        _get_subject_spectral_subdir(
            subject,
            "tfr",
        )
    )

    return (
        output_directory
        / f"{channel}.png"
    )


def get_psd_path(
    subject: int,
    channel: str,
) -> Path:
    """
    Return the power spectral density figure path.
    """
    output_directory = (
        _get_subject_spectral_subdir(
            subject,
            "psd",
        )
    )

    return (
        output_directory
        / f"{channel}.png"
    )


# ----------------------------------------------------------------------
# SHAP result paths
# ----------------------------------------------------------------------

def get_shap_values_path(
    subject: int,
) -> Path:
    """
    Return the saved SHAP values path for one subject.
    """
    return _get_relevance_values_path(
        method="shap",
        subject=subject,
    )


def get_shap_channel_time_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP channel-time plot path.
    """
    return _get_relevance_plot_path(
        method="shap",
        plot_type="channel_time",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_temporal_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP temporal plot path.
    """
    return _get_relevance_plot_path(
        method="shap",
        plot_type="temporal_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_topographies_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP topography plot path.
    """
    return _get_relevance_plot_path(
        method="shap",
        plot_type="topographies",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_channel_rankings_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP channel-ranking plot path.
    """
    return _get_relevance_plot_path(
        method="shap",
        plot_type="channel_rankings",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_channel_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP channel-relevance plot path.
    """
    return _get_relevance_plot_path(
        method="shap",
        plot_type="channel_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


# ----------------------------------------------------------------------
# CSP occlusion result paths
# ----------------------------------------------------------------------

def get_csp_values_path(
    subject: int,
) -> Path:
    """
    Return the saved CSP occlusion values path for one subject.
    """
    return _get_relevance_values_path(
        method="csp",
        subject=subject,
    )


def get_csp_channel_time_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean CSP channel-time plot path.
    """
    return _get_relevance_plot_path(
        method="csp",
        plot_type="channel_time",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_csp_temporal_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean CSP temporal plot path.
    """
    return _get_relevance_plot_path(
        method="csp",
        plot_type="temporal_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_csp_topographies_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean CSP topography plot path.
    """
    return _get_relevance_plot_path(
        method="csp",
        plot_type="topographies",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_csp_channel_rankings_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean CSP channel-ranking plot path.
    """
    return _get_relevance_plot_path(
        method="csp",
        plot_type="channel_rankings",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_csp_channel_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean CSP channel-relevance plot path.
    """
    return _get_relevance_plot_path(
        method="csp",
        plot_type="channel_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_csp_pattern_values_path(
    subject: int,
) -> Path:
    """
    Return the saved CSP spatial-pattern values path.
    """
    subject_name = get_subject_name(
        subject
    )

    output_dir = _create_directory(
        CSP_PATTERN_RESULTS_DIR / "values"
    )

    return (
        output_dir
        / f"{subject_name}_csp_patterns.npz"
    )


def get_csp_subject_patterns_path(
    subject: int,
) -> Path:
    """
    Return the CSP spatial-pattern figure path for one subject.
    """
    subject_name = get_subject_name(
        subject
    )

    output_dir = _create_directory(
        CSP_PATTERN_RESULTS_DIR
        / "subject_patterns"
    )

    return (
        output_dir
        / f"{subject_name}_csp_patterns.png"
    )