import os
import sys
import subprocess

from fastapi.testclient import TestClient

# --- rendre "src" importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.api import app  # noqa: E402


ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "src", "mlops_tp", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "src", "train.py")


def ensure_model_exists():
    """Assure que model.joblib existe avant de lancer les tests API."""
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
                "car_ID": 1,
                "symboling": 0,
                "CarName": "test car",
                "fueltype": "gas",
                "aspiration": "std",
                "doornumber": "two",
                "carbody": "hatchback",
                "drivewheel": "fwd",
                "enginelocation": "front",
                "wheelbase": 90.0,
                "carlength": 150.0,
                "carwidth": 60.0,
                "carheight": 50.0,
                "curbweight": 2000,
                "enginetype": "ohc",
                "cylindernumber": "four",
                "enginesize": 100,
                "fuelsystem": "mpfi",
                "boreratio": 3.0,
                "stroke": 3.0,
                "compressionratio": 9.0,
                "horsepower": 90,
                "peakrpm": 5000,
                "citympg": 25,
                "highwaympg": 30
            }
        }

        response = client.post("/predict", json=payload)

        # si ça échoue encore, tu verras le détail
        assert response.status_code == 200, response.text
        assert "prediction" in response.json()


def test_predict_validation_error():
    ensure_model_exists()
    with TestClient(app) as client:
        response = client.post("/predict", json={})
        assert response.status_code == 422