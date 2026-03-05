from pathlib import Path

# Dossier du package mlops_tp
PKG_DIR = Path(__file__).resolve().parent

# Artifacts dans mlops_tp/artifacts
ARTIFACTS_DIR = PKG_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
RUN_INFO_PATH = ARTIFACTS_DIR / "run_info.json"

# Dataset (dans /data à la racine)
PROJECT_ROOT = PKG_DIR.parent.parent   # .../TP_MLOPPS
DATA_PATH = PROJECT_ROOT / "data" / "CarPrice_Assignment.csv"

TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42