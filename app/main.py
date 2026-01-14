from fastapi import FastAPI, HTTPException
import pandas as pd
import math
from app.schemas import TransactionRequest
from app.model_loader import ModelService
from app.logger import logger

# Reuse existing feature engineering
from src.feature_engineering.preprocess import build_features


app = FastAPI(
    title="Fraud Anomaly Detection API",
    description="Real-time fraud detection using Isolation Forest with online feature engineering",
    version="1.0.0",
)

model_service = ModelService()

def score_to_probability(score: float, threshold: float) -> float:
    # distance from decision boundary
    margin = score - threshold

    # sigmoid mapping
    prob = 1 / (1 + math.exp(-10 * margin))
    return round(prob, 4)

# Health check
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API is running"}


# Prediction endpoint
@app.post("/predict")
def predict_fraud(request: TransactionRequest):
    try:
        logger.info(f"Incoming transaction: {request}")

        # Raw input → DataFrame
        raw_df = pd.DataFrame([request.dict()])

        #Feature engineering (same as training)
        features_df = build_features(raw_df)

        #One-row → dict
        features_dict = features_df.iloc[0].to_dict()

        #Model prediction
        score = model_service.predict(features_dict)
        fraud_probability = score_to_probability(# Convert score → probability
            score,
            model_service.threshold 
        )
        is_fraud = fraud_probability >= 0.5

        #MONITORING / DEBUG LOG (ADD THIS)
        logger.info(
            f"SCORE_DEBUG | "
            f"score={round(score, 4)} | "
            f"threshold={round(model_service.threshold, 4)} | "
            f"fraud_probability={fraud_probability} | "
            f"is_fraud={is_fraud}"
        )

        response = {
            "fraud_probability": round(float(fraud_probability), 4),
            "fraud_score": round(float(score), 4),
            "is_fraud": bool(is_fraud),
            "explanation": (
                "Transaction shows anomalous behavior compared to customer's, historical spending patterns (amount/velocity/location deviation)."
            if is_fraud
                else "Transaction falls within the customer's normal behavioral range "
                        "based on historical patterns."
            ),
        }


        logger.info(f"Prediction response: {response}")
        return response

    except ValueError as ve:
        logger.error(str(ve))
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction",
        )
