from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

EEGNET_MODEL_DIR = MODEL_DIR / "eegnet"
CSP_LDA_MODEL_DIR = MODEL_DIR / "csp_lda"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EEGNET_MODEL_DIR.mkdir(parents=True, exist_ok=True)
CSP_LDA_MODEL_DIR.mkdir(parents=True, exist_ok=True)