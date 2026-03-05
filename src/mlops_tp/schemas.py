# src/mlops_tp/schemas.py

from pydantic import BaseModel
from typing import Dict, Any


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str