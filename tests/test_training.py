
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "train.py")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "src", "mlops_tp", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_schema.json")


def test_training_end_to_end_creates_artifacts():
    # Lance l'entraînement (même python que pytest)
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    # Vérifie la génération des artefacts demandés
    assert os.path.exists(MODEL_PATH), f"Missing artifact: {MODEL_PATH}"
    assert os.path.exists(METRICS_PATH), f"Missing artifact: {METRICS_PATH}"
    assert os.path.exists(SCHEMA_PATH), f"Missing artifact: {SCHEMA_PATH}"