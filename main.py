from pathlib import Path
import pickle
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Train_model" / "model_bundle.pkl"

app = FastAPI(
    title="Loan Payback Prediction API",
    description="Demo API that returns a model prediction and estimated repayment probability.",
    version="0.1.0",
)


class LoanFeatures(BaseModel):
    age: float
    gender: str
    marital_status: str
    education_level: str
    annual_income: float
    monthly_income: float
    employment_status: str
    debt_to_income_ratio: float
    credit_score: float
    loan_amount: float
    loan_purpose: str
    interest_rate: float
    loan_term: float
    installment: float
    grade_subgrade: str
    num_of_open_accounts: float
    total_credit_limit: float
    current_balance: float
    delinquency_history: float
    public_records: float
    num_of_delinquencies: float


class PredictionResponse(BaseModel):
    prediction: str
    probability_repaid: float
    threshold: float
    model: str


def load_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    # Only load pickle files created by this project; untrusted pickle files can execute code.
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


model_bundle = load_bundle()


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
    }


def _predict(values: dict[str, Any]) -> PredictionResponse:
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run the notebook's model export cell first.",
        )

    input_frame = pd.DataFrame([values])
    feature_columns = model_bundle["feature_columns"]
    categorical_columns = model_bundle["categorical_columns"]

    missing_columns = set(feature_columns) - set(input_frame.columns)
    if missing_columns:
        raise HTTPException(status_code=500, detail=f"Missing model features: {sorted(missing_columns)}")

    encoded_categories = pd.DataFrame(
        model_bundle["encoder"].transform(input_frame[categorical_columns]),
        columns=categorical_columns,
        index=input_frame.index,
    )
    processed = input_frame.drop(columns=categorical_columns).join(encoded_categories)
    processed = processed.reindex(columns=feature_columns)
    scaled = model_bundle["scaler"].transform(processed)

    model = model_bundle["model"]
    probabilities = model.predict_proba(scaled)[0]
    class_one_index = list(model.classes_).index(1)
    probability_repaid = float(probabilities[class_one_index])
    threshold = float(model_bundle["threshold"])
    prediction = "repaid" if probability_repaid >= threshold else "not_repaid"

    return PredictionResponse(
        prediction=prediction,
        probability_repaid=probability_repaid,
        threshold=threshold,
        model="tuned AdaBoost",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: LoanFeatures) -> PredictionResponse:
    # Keep CPU-bound pandas and model inference off the async event loop.
    return await run_in_threadpool(_predict, features.model_dump())