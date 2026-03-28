import os
import sys
import subprocess

from fastapi.testclient import TestClient

# --- rendre "src" importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.mlops_tp.api import app

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "src", "mlops_tp", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "mlops_tp", "train.py")


def ensure_model_exists():
    if os.path.exists(MODEL_PATH):
        return
    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, (
        f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert os.path.exists(MODEL_PATH), "model.joblib was not created"


def test_health():
    ensure_model_exists()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_success():
    ensure_model_exists()
    with TestClient(app) as client:
        payload = {
            "features": {
                "model": "5 Series",
                "year": 2018,
                "transmission": "Automatic",
                "mileage": 50000,
                "fuelType": "Diesel",
                "tax": 145,
                "mpg": 50.4,
                "engineSize": 2.0
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        assert "prediction" in response.json()


def test_predict_validation_error():
    ensure_model_exists()
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422