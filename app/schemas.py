from datetime import datetime
from pydantic import BaseModel, Field
from typing import Dict

class TransactionRequest(BaseModel):
    customer_id: str
    amount: float
    timestamp: datetime
    hour: int
    distance_from_home: float

class PredictionResponse(BaseModel):
    fraud_score: float
    is_fraud: bool
    explanation: str
