from pathlib import Path

# Chemin vers le dossier mlops_tp
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = BASE_DIR.parent.parent / "data" / "CarPrice_Assignment.csv"

# Artifacts
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"
RUN_INFO_PATH = ARTIFACTS_DIR / "run_info.json"

TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42