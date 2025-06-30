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
    try:
        print("✅ Request received:", request.dict())
        
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        print("✅ Input DataFrame:\n", input_data.head())
        
        # Predict
        proba = model.predict_proba(input_data)[0][1]
        prediction = (proba > 0.5).astype(int)
        
        print(f"✅ Prediction: {prediction}, Probability: {proba}")
        
        return {
            "is_high_risk": int(prediction),
            "risk_probability": float(proba)
        }
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))