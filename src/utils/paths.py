from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

EEGNET_MODEL_DIR = MODEL_DIR / "eegnet"
CSP_LDA_MODEL_DIR = MODEL_DIR / "csp_lda"
ACCURACY_RESULTS_DIR = RESULTS_DIR / "accuracies"

ACCURACY_COMPARISON_FILE = ACCURACY_RESULTS_DIR / "csp_lda_eegnet.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EEGNET_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSP_LDA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ACCURACY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)