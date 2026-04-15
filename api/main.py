# api/main.py
import pandas as pd
import numpy as np
import mlflow
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from api.schemas import LoanApplication, PredictionResponse, HealthResponse
from api.model_loader import model_loader
from src.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ALL_FEATURE_COLUMNS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'AMT_INCOME_TOTAL',
    'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE', 'REGION_RATING_CLIENT', 'DAYS_BIRTH',
    'DAYS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
    'DAYS_LAST_PHONE_CHANGE', 'AMT_REQ_CREDIT_BUREAU_YEAR'
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Loading model and preprocessor...")
    model_loader.load()
    logger.info("API ready to serve predictions")
    yield
    logger.info("Shutting down API")


app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts probability of loan default using XGBoost trained on Home Credit data.",
    version="1.0.0",
    lifespan=lifespan
)


def probability_to_score(probability: float) -> int:
    """
    Convert default probability to a credit score (300-850).
    Higher score = lower risk = less likely to default.
    Standard scorecard transformation used in credit risk.
    """
    import math
    if probability <= 0:
        probability = 0.0001
    if probability >= 1:
        probability = 0.9999
    score = 850 - int(550 * math.log(probability / (1 - probability) + 10) / math.log(10))
    return max(300, min(850, score))


def classify_risk(probability: float) -> str:
    """Classify risk into low, medium, high bands."""
    if probability < 0.10:
        return "low"
    elif probability < 0.25:
        return "medium"
    else:
        return "high"


def prepare_input(application: LoanApplication) -> pd.DataFrame:
    """Convert the input schema into a dataframe ready for the preprocessor."""
    import joblib
    from pathlib import Path

    # Load the feature names the preprocessor was trained on
    preprocessor = model_loader.preprocessor
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    all_features = num_features + cat_features

    # Start with the submitted data
    data = application.model_dump()
    df = pd.DataFrame([data])

    # Add all missing columns as NaN
    for col in all_features:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder to match training order exactly
    df = df[all_features]
    return df


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and model health."""
    return HealthResponse(
        status="healthy" if model_loader.is_loaded else "degraded",
        model_loaded=model_loader.is_loaded,
        preprocessor_loaded=model_loader.is_loaded,
        mlflow_version=mlflow.__version__
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(application: LoanApplication):
    """
    Predict the probability of loan default for a given application.

    Returns:
    - probability_of_default: float between 0 and 1
    - risk_class: low / medium / high
    - risk_score: credit score 300-850
    - model_version: which model was used
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = prepare_input(application)
        processed = model_loader.preprocessor.transform(input_df)
        probability = float(model_loader.model.predict_proba(processed)[0][1])

        response = PredictionResponse(
            probability_of_default=round(probability, 4),
            risk_class=classify_risk(probability),
            risk_score=probability_to_score(probability),
            model_version="xgboost_v1"
        )

        logger.info(f"Prediction: prob={probability:.4f}, risk={response.risk_class}, score={response.risk_score}")
        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    """API root — confirms service is running."""
    return {
        "service": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }