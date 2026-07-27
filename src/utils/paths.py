from pathlib import Path

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
ERD_RESULTS_DIR = RESULTS_DIR / "erd"


# Create output directories
EEGNET_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSP_LDA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
ACCURACY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_MATRIX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ERD_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_subject_name(subject: int) -> str:
    """Return the formatted subject name."""
    return f"A{subject:02d}"


def get_subject_files(
    subject: int,
) -> tuple[Path, Path, Path] | None:
    """Return the data files for one subject."""
    subject_name = get_subject_name(subject)

    train_file = DATA_DIR / f"{subject_name}T.gdf"
    eval_file = DATA_DIR / f"{subject_name}E.gdf"
    mat_file = DATA_DIR / f"{subject_name}E.mat"

    if not all(
        path.exists()
        for path in (train_file, eval_file, mat_file)
    ):
        return None

    return train_file, eval_file, mat_file


def is_eval_file(file_path: str | Path) -> bool:
    """Check whether a file belongs to the evaluation session."""
    return Path(file_path).stem.upper().endswith("E")


def get_csp_lda_subject_dir(subject: int) -> Path:
    """Return the CSP+LDA directory for one subject."""
    path = (
        CSP_LDA_MODEL_DIR
        / f"seed_{BASE_SEED}"
        / get_subject_name(subject)
    )

    path.mkdir(parents=True, exist_ok=True)

    return path


def get_csp_fold_model_path(
    subject: int,
    fold: int,
) -> Path:
    """Return the CSP model path for one fold."""
    subject_name = get_subject_name(subject)

    filename = (
        f"{subject_name}_csp_kfold_"
        f"seed{BASE_SEED}_fold{fold}.joblib"
    )

    return get_csp_lda_subject_dir(subject) / filename


def get_lda_fold_model_path(
    subject: int,
    fold: int,
) -> Path:
    """Return the LDA model path for one fold."""
    subject_name = get_subject_name(subject)

    filename = (
        f"{subject_name}_lda_kfold_"
        f"seed{BASE_SEED}_fold{fold}.joblib"
    )

    return get_csp_lda_subject_dir(subject) / filename


def get_eegnet_subject_dir(subject: int) -> Path:
    """Return the EEGNet directory for one subject."""
    path = (
        EEGNET_MODEL_DIR
        / f"seed_{BASE_SEED}"
        / get_subject_name(subject)
    )

    path.mkdir(parents=True, exist_ok=True)

    return path


def get_eegnet_fold_model_path(
    subject: int,
    fold: int,
) -> Path:
    """Return the EEGNet model path for one fold."""
    subject_name = get_subject_name(subject)

    filename = (
        f"{subject_name}_eegnet_kfold_"
        f"seed{BASE_SEED}_fold{fold}.keras"
    )

    return get_eegnet_subject_dir(subject) / filename


def get_results_accuracy_comparison_path() -> Path:
    """Return the accuracy comparison CSV path."""
    return (
        ACCURACY_RESULTS_DIR
        / f"seed_{BASE_SEED}_csp_lda_vs_eegnet.csv"
    )

def get_all_confusion_matrices_path() -> Path:
    """Return the all-subject confusion matrices png path."""
    path = (
        CONFUSION_MATRIX_RESULTS_DIR
        / f"seed_{BASE_SEED}"
    )

    path.mkdir(parents=True, exist_ok=True)

    return (
        path
        / f"seed_{BASE_SEED}_all_confusion_matrices.png"
    )

def get_subject_confusion_matrices_path(subject: int) -> Path:
    """Return the confusion matrices png path for one subject."""
    subject_name = get_subject_name(subject)

    path = (
        CONFUSION_MATRIX_RESULTS_DIR
        / f"seed_{BASE_SEED}"
    )

    path.mkdir(parents=True, exist_ok=True)

    return (
        path
        / f"seed_{BASE_SEED}_{subject_name}_confusion_matrices.png"
    )