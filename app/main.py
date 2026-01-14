from fastapi import FastAPI, HTTPException
from app.schemas import TransactionRequest, PredictionResponse
from app.model_loader import ModelService
from app.logger import logger

app = FastAPI(
    title="Fraud Detection API",
    description="Anomaly detection using Isolation Forest",
    version="1.0"
)

model_service = ModelService()

# Threshold chosen from your analysis (1% operating point)
ANOMALY_THRESHOLD = 0.225931  

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(request: TransactionRequest):
    try:
        logger.info(f"Received request: {request.features}")

        score = model_service.predict(request.features)
        is_fraud = score >= ANOMALY_THRESHOLD

        response = PredictionResponse(
            fraud_score=round(score, 4),
            is_fraud=is_fraud,
            explanation=(
                "Transaction deviates significantly from normal behavior"
                if is_fraud else
                "Transaction appears consistent with normal patterns"
            )
        )

        logger.info(f"Prediction result: {response}")
        return response

    except ValueError as ve:
        logger.error(str(ve))
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )
