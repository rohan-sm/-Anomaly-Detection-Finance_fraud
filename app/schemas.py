from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    customer_id: str = Field(
        ..., min_length=1, examples=["CUST_00042"]
    )
    amount: float = Field(
        ..., gt=0, description="Transaction amount in INR"
    )
    timestamp: datetime = Field(
        ..., description="ISO-8601 transaction time"
    )
    hour: int = Field(
        ..., ge=0, le=23, description="Local hour of day, 0-23"
    )
    distance_from_home: float = Field(
        ..., ge=0, description="Haversine distance from customer home in km"
    )


class FeatureContribution(BaseModel):
    """One feature's contribution to the decision.

    `z_score` is how many standard deviations the observed value sits from the
    population mean captured by the training scaler, i.e. how far this feature
    sits outside the range the model was fitted on.
    """

    feature: str
    value: float
    z_score: float
    direction: str = Field(
        ..., description="'above_normal' or 'below_normal' relative to population"
    )


class PredictionResponse(BaseModel):
    fraud_score: float = Field(
        ..., description="Raw Isolation Forest anomaly score (higher = more anomalous)"
    )
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Sigmoid-mapped score around the decision threshold"
    )
    is_fraud: bool
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")
    contributing_factors: List[FeatureContribution]
    explanation: str


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(
        ..., min_length=1, max_length=1000
    )


class BatchPredictionResponse(BaseModel):
    count: int
    flagged: int
    flag_rate: float
    mean_fraud_probability: float
    predictions: List[PredictionResponse]
