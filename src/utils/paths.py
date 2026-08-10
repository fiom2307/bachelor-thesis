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
CSP_PATTERN_ANALYSIS_DIR = RESULTS_DIR / "csp_pattern_analysis"


TrialSelection = Literal[
    "correct",
    "incorrect",
]

SHAPDomain = Literal[
    "time_domain",
    "frequency_domain",
]

SHAPPlotType = Literal[
    "temporal_relevance",
    "frequency_relevance",
    "topographies",
    "channel_rankings",
    "channel_relevance",
]

CSPPatternPlotType = Literal[
    "channel_rankings",
    "channel_relevance",
    "topographies",
    "temporal_relevance",
    "frequency_relevance",
]


# Create main output directories
for directory in (
    EEGNET_MODEL_DIR,
    CSP_LDA_MODEL_DIR,
    ACCURACY_RESULTS_DIR,
    CONFUSION_MATRIX_RESULTS_DIR,
    SPECTRAL_ANALYSIS_DIR,
    SHAP_RESULTS_DIR,
    CSP_PATTERN_ANALYSIS_DIR,
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


def _get_shap_result_name(
    subject: int | None,
) -> str:
    """
    Return the filename prefix for a subject-wise or global mean result.
    """
    if subject is None:
        return "mean"

    return get_subject_name(subject)


def _get_shap_plot_path(
    plot_type: SHAPPlotType,
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP plot path.
    """
    output_directory = _create_directory(
        SHAP_RESULTS_DIR / plot_type
    )

    result_name = _get_shap_result_name(
        subject
    )

    filename = (
        f"{result_name}_"
        f"shap_"
        f"{plot_type}_"
        f"{trial_selection}.png"
    )

    return output_directory / filename


def _get_shap_values_path(
    subject: int,
    domain: SHAPDomain,
) -> Path:
    """
    Return the saved SHAP values path for one subject and domain.
    """
    subject_name = get_subject_name(
        subject
    )

    output_directory = _create_directory(
        SHAP_RESULTS_DIR / "shap_values"
    )

    return (
        output_directory
        / f"{subject_name}_{domain}.npz"
    )


def _get_csp_pattern_result_name(
    subject: int | None,
) -> str:
    """
    Return the filename prefix for a subject-wise
    or global mean CSP pattern result.
    """
    if subject is None:
        return "all_mean"

    return get_subject_name(subject)


def _get_csp_pattern_plot_path(
    plot_type: CSPPatternPlotType,
    subject: int | None,
) -> Path:
    """
    Return a subject-wise or global mean CSP pattern plot path.
    """
    output_directory = _create_directory(
        CSP_PATTERN_ANALYSIS_DIR / plot_type
    )

    result_name = _get_csp_pattern_result_name(
        subject
    )

    filename = (
        f"{result_name}_"
        f"csp_"
        f"{plot_type}.png"
    )

    return output_directory / filename


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

def get_time_domain_shap_values_path(
    subject: int,
) -> Path:
    """
    Return the saved time-domain SHAP values path for one subject.
    """
    return _get_shap_values_path(
        subject=subject,
        domain="time_domain",
    )


def get_frequency_domain_shap_values_path(
    subject: int,
) -> Path:
    """
    Return the saved frequency-domain SHAP values path for one subject.
    """
    return _get_shap_values_path(
        subject=subject,
        domain="frequency_domain",
    )


def get_shap_temporal_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP temporal-relevance path.
    """
    return _get_shap_plot_path(
        plot_type="temporal_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_frequency_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP frequency-relevance path.
    """
    return _get_shap_plot_path(
        plot_type="frequency_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_topographies_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP topography path.
    """
    return _get_shap_plot_path(
        plot_type="topographies",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_channel_rankings_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP channel-ranking path.
    """
    return _get_shap_plot_path(
        plot_type="channel_rankings",
        subject=subject,
        trial_selection=trial_selection,
    )


def get_shap_channel_relevance_path(
    subject: int | None,
    trial_selection: TrialSelection,
) -> Path:
    """
    Return a subject-wise or global mean SHAP channel-relevance path.
    """
    return _get_shap_plot_path(
        plot_type="channel_relevance",
        subject=subject,
        trial_selection=trial_selection,
    )


# ----------------------------------------------------------------------
# CSP pattern analysis paths
# ----------------------------------------------------------------------

def get_csp_channel_relevance_path(
    subject: int | None,
) -> Path:
    """
    Return a subject-wise or global mean
    CSP channel-relevance path.
    """
    return _get_csp_pattern_plot_path(
        plot_type="channel_relevance",
        subject=subject,
    )


def get_csp_channel_rankings_path(
    subject: int | None,
) -> Path:
    """
    Return a subject-wise or global mean
    CSP channel-ranking path.
    """
    return _get_csp_pattern_plot_path(
        plot_type="channel_rankings",
        subject=subject,
    )


def get_csp_temporal_relevance_path(
    subject: int | None,
) -> Path:
    """
    Return a subject-wise or global mean
    CSP temporal-relevance path.
    """
    return _get_csp_pattern_plot_path(
        plot_type="temporal_relevance",
        subject=subject,
    )


def get_csp_frequency_relevance_path(
    subject: int | None,
) -> Path:
    """
    Return a subject-wise or global mean
    CSP frequency-relevance path.
    """
    return _get_csp_pattern_plot_path(
        plot_type="frequency_relevance",
        subject=subject,
    )


def get_csp_topographies_path(
    subject: int | None,
) -> Path:
    """
    Return a subject-wise or global mean
    CSP topographies path.
    """
    return _get_csp_pattern_plot_path(
        plot_type="topographies",
        subject=subject,
    )