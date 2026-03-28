import os
import json
import time
from datetime import datetime
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.mlops_tp.schemas import PredictionRequest, PredictionResponse, HealthResponse

# =========================================================
# Chemins
# =========================================================
BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, "feature_schema.json")

# =========================================================
# App
# =========================================================
app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0",
    description="API REST FastAPI pour prédire le prix d'une voiture à partir de ses caractéristiques."
)

model = None
feature_schema: Dict[str, Any] | None = None

@app.get("/")
def root():
    return {"message": "Car Price Prediction API", "docs": "/docs", "health": "/health"}

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

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok - version 2"}

@app.get("/metadata")
def metadata():
    if feature_schema is None:
        return {"task": "regression", "model": "loaded", "message": "feature_schema.json not found"}
    return {
        "task": "regression",
        "model": "RandomForest (pipeline)",
        "dataset": feature_schema.get("dataset"),
        "target": feature_schema.get("target"),
        "n_features": feature_schema.get("n_features"),
        "features": feature_schema.get("features"),
        "api_version": app.version,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
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
    return {"prediction": value}