from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple


class PredictionSchema(BaseModel):
    disease_name: str  # Friendly disease name
    confidence: float
    crop: str
    healthy: bool


class AdviceSchema(BaseModel):
    description: str
    treatment: str
    prevention: str
    urgency: str


class PredictionResponse(BaseModel):
    predictions: List[PredictionSchema]
    advice: AdviceSchema
    note: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    repo_id: str