# src/api/main.py

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Credit Risk API", description="Predict customer risk using trained model", version="1.0")

# Load best model
model = joblib.load("models/random_forest_best.pkl")
print("✅ Model loaded")


class CreditRiskRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    DaysSinceFirstTransaction: int
    TotalTransactionAmount: float
    AvgTransactionAmount: float
    TransactionCount: int
    StdTransactionAmount: float
    MinTransactionAmount: float
    MaxTransactionAmount: float
    NumUniqueProducts: int
    NumUniqueChannels: int


class CreditRiskResponse(BaseModel):
    is_high_risk: int
    risk_probability: float


@app.post("/predict", response_model=CreditRiskResponse)
def predict(request: CreditRiskRequest):
    """
    Predict if a customer is high-risk based on transaction data
    """
    # Convert request to DataFrame
    input_data = pd.DataFrame([request.dict()])
    
    # Predict
    proba = model.predict_proba(input_data)[0][1]
    prediction = (proba > 0.5).astype(int)

    return {
        "is_high_risk": int(prediction),
        "risk_probability": float(proba)
    }