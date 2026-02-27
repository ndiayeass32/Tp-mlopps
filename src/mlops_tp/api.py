import os
import json
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# =========================================================
# Chemins
# =========================================================
# =========================================================
# Chemins (robustes)
# =========================================================
BASE_DIR = os.path.dirname(__file__)                # .../src/mlops_tp
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts") # .../src/mlops_tp/artifacts

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_schema.json")


# =========================================================
# Chargement au démarrage
# =========================================================
app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0",
    description="API REST FastAPI pour prédire le prix d'une voiture à partir de ses caractéristiques."
)

model = None
feature_schema: Dict[str, Any] | None = None


@app.on_event("startup")
def load_artifacts():
    global model, feature_schema

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            feature_schema = json.load(f)
    else:
        feature_schema = None


# =========================================================
# Pydantic schema d'entrée (générique)
# On accepte un dict "features" => plus simple et robuste
# =========================================================
class PredictRequest(BaseModel):
    features: Dict[str, Any]


# =========================================================
# 1) GET /health
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


# =========================================================
# 2) GET /metadata
# =========================================================
@app.get("/metadata")
def metadata():
    if feature_schema is None:
        return {
            "task": "regression",
            "model": "loaded",
            "message": "feature_schema.json not found"
        }

    return {
        "task": "regression",
        "model": "Ridge (pipeline)",
        "dataset": feature_schema.get("dataset"),
        "target": feature_schema.get("target"),
        "n_features": feature_schema.get("n_features"),
        "features": feature_schema.get("features"),
        "api_version": app.version,
    }


# =========================================================
# 3) POST /predict
# =========================================================
import time

@app.post("/predict")
def predict(payload: PredictRequest):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    start = time.time()

    X = pd.DataFrame([payload.features])

    try:
        pred = model.predict(X)
        value = float(pred[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    latency = (time.time() - start) * 1000

    return {
        "prediction": value,
        "task": "regression",
        "model_version": app.version,
        "latency_ms": round(latency, 3)
    }