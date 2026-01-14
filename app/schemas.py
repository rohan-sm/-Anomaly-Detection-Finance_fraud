from pydantic import BaseModel, Field
from typing import Dict

class TransactionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., 
        example={"amount": 5000.0, "hour": 14, "distance_from_home": 12.5}
    )

class PredictionResponse(BaseModel):
    fraud_score: float
    is_fraud: bool
    explanation: str
