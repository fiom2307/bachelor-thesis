from pathlib import Path

from src.utils.config import BASE_SEED

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

EEGNET_MODEL_DIR = MODEL_DIR / "eegnet"
CSP_LDA_MODEL_DIR = MODEL_DIR / "csp_lda"
ACCURACY_RESULTS_DIR = RESULTS_DIR / "accuracies"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EEGNET_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSP_LDA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ACCURACY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def get_subject_files(subj: int):
    train_file  = DATA_DIR / f"A0{subj}T.gdf"
    eval_file  = DATA_DIR / f"A0{subj}E.gdf"
    mat_file = DATA_DIR / f"A0{subj}E.mat"

    if not (train_file.exists() and eval_file.exists() and mat_file.exists()):
        return None

    return train_file, eval_file, mat_file

def is_eval_file(file_path: Path):
    file_stem = file_path.stem
    session = file_stem[3]
    return session.upper() == "E"

def get_subject_name(subject: int):
    return f"A{subject:02d}"


def get_csp_lda_subject_dir(subject: int):
    subject_name = get_subject_name(subject)
    path = CSP_LDA_MODEL_DIR / f"seed_{BASE_SEED}" / subject_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_csp_fold_model_path(subject: int, fold: int):
    return get_csp_lda_subject_dir(subject) / f"A{subject:02d}_csp_kfold_seed{BASE_SEED}_fold{fold}.joblib"

def get_lda_fold_model_path(subject: int, fold: int):
    return get_csp_lda_subject_dir(subject) / f"A{subject:02d}_lda_kfold_seed{BASE_SEED}_fold{fold}.joblib"


def get_eegnet_seed_subject_dir(subject: int):
    subject_name = get_subject_name(subject)
    path = EEGNET_MODEL_DIR / f"seed_{BASE_SEED}" / subject_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_eegnet_fold_model_path(subject, fold):
    return get_eegnet_seed_subject_dir(subject) / f"A{subject:02d}_eegnet_kfold_seed{BASE_SEED}_fold{fold}.keras"

def get_results_accuracy_comparison_path():
    return ACCURACY_RESULTS_DIR / f"seed_{BASE_SEED}_csp_lda_vs_eegnet.csv"