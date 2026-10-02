import math

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.logger import logger
from app.model_loader import ModelService
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.feature_engineering.preprocess import build_features

# Steepness of the sigmoid that converts an anomaly score into a probability.
# Fixed so that a score sitting exactly on the threshold maps to 0.5.
PROBABILITY_STEEPNESS = 10.0

HIGH_RISK_CUTOFF = 0.8
FLAG_CUTOFF = 0.5

# Internal column used to restore submission order after feature engineering
# re-sorts the frame by customer.
ORDER_KEY = "_request_order"

app = FastAPI(
    title="Fraud Anomaly Detection API",
    description=(
        "Real-time fraud detection using Isolation Forest with online feature "
        "engineering. Every prediction returns a probability, a risk band, and "
        "the features that drove the decision."
    ),
    version="1.1.0",
)

model_service = ModelService()


def score_to_probability(score: float, threshold: float) -> float:
    """Map an anomaly score onto [0, 1] around the decision threshold.

    Isolation Forest emits a score, not a probability. Measuring the signed
    distance from the configured threshold and passing it through a steep
    sigmoid keeps the threshold itself at exactly 0.5, so the decision rule and
    the probability always agree.
    """
    margin = score - threshold
    prob = 1 / (1 + math.exp(-PROBABILITY_STEEPNESS * margin))
    return round(prob, 4)


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK_CUTOFF:
        return "HIGH"
    if probability >= FLAG_CUTOFF:
        return "MEDIUM"
    return "LOW"


def build_explanation(is_fraud: bool, factors: list[dict]) -> str:
    detail = ", ".join(
        f"{f['feature']}={f['value']} (z={f['z_score']:+.1f})" for f in factors
    )
    if not factors:
        return "No features available for explanation."

    if is_fraud:
        return (
            "Transaction deviates from the learned population on: "
            f"{detail}. Strongest driver: {factors[0]['feature']} is "
            f"{factors[0]['direction'].replace('_', ' ')} the norm."
        )

    return (
        "Transaction sits inside the normal behavioural range. Closest signals: "
        f"{detail}. None deviated far enough to trigger the alert threshold."
    )


def score_transactions(raw_df: pd.DataFrame) -> list[dict]:
    """Run the full serving path over a raw transaction frame.

    Feature engineering reuses `build_features`, the exact function used to
    build the training matrix, so there is no training-serving skew.

    `build_features` sorts by customer to make the rolling windows correct,
    which permutes the rows. An explicit order key is attached beforehand and
    undone afterwards so predictions always come back in submission order.
    """
    raw_df = raw_df.copy()
    raw_df[ORDER_KEY] = range(len(raw_df))

    features_df = build_features(raw_df)
    features_df = features_df.sort_values(ORDER_KEY).reset_index(drop=True)

    missing = [
        feature
        for feature in model_service.features
        if feature not in features_df.columns
    ]
    if missing:
        raise ValueError(f"Feature engineering did not produce: {missing}")

    features_rows = features_df[model_service.features].to_dict("records")
    scores = model_service.predict_batch(features_rows)

    predictions = []
    for row, score in zip(features_rows, scores):
        score = float(score)
        probability = score_to_probability(score, model_service.threshold)
        is_fraud = probability >= FLAG_CUTOFF
        factors = model_service.feature_contributions(row)

        predictions.append(
            {
                "fraud_score": round(score, 4),
                "fraud_probability": probability,
                "is_fraud": is_fraud,
                "risk_level": risk_level(probability),
                "contributing_factors": factors,
                "explanation": build_explanation(is_fraud, factors),
            }
        )

    return predictions


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Fraud Detection API is running",
        "model": "isolation_forest_v1",
        "features": len(model_service.features),
    }


@app.get("/health")
def health_probe():
    """Liveness/readiness probe for container orchestration."""
    return {"status": "healthy", "threshold": model_service.threshold}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: TransactionRequest):
    try:
        logger.info(f"Incoming transaction: {request}")

        raw_df = pd.DataFrame([request.model_dump()])
        prediction = score_transactions(raw_df)[0]

        logger.info(
            f"SCORE_DEBUG | "
            f"score={prediction['fraud_score']} | "
            f"threshold={round(model_service.threshold, 4)} | "
            f"fraud_probability={prediction['fraud_probability']} | "
            f"risk_level={prediction['risk_level']} | "
            f"is_fraud={prediction['is_fraud']} | "
            f"top_factor={prediction['contributing_factors'][0]['feature']}"
        )

        return prediction

    except ValueError as ve:
        logger.error(str(ve))
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction",
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def predict_fraud_batch(request: BatchPredictionRequest):
    """Score many transactions in one vectorised pass.

    Per-request scoring rebuilds a one-row DataFrame and runs a single
    `decision_function` call, so its cost is dominated by fixed overhead.
    Batching amortises that across the whole request.
    """
    try:
        raw_df = pd.DataFrame([t.model_dump() for t in request.transactions])
        predictions = score_transactions(raw_df)

        flagged = sum(1 for p in predictions if p["is_fraud"])
        probabilities = [p["fraud_probability"] for p in predictions]

        logger.info(
            f"BATCH_DEBUG | n={len(predictions)} | flagged={flagged} | "
            f"flag_rate={flagged / len(predictions):.4f}"
        )

        return {
            "count": len(predictions),
            "flagged": flagged,
            "flag_rate": round(flagged / len(predictions), 4),
            "mean_fraud_probability": round(sum(probabilities) / len(probabilities), 4),
            "predictions": predictions,
        }

    except ValueError as ve:
        logger.error(str(ve))
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during batch prediction",
        )
