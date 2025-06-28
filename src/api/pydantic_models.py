# src/api/pydantic_models.py

from pydantic import BaseModel
from typing import Optional

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

    class Config:
        schema_extra = {
            "example": {
                "TransactionId": "TransactionId_34164",
                "BatchId": "BatchId_130819",
                "AccountId": "AccountId_3210",
                "SubscriptionId": "SubscriptionId_3348",
                "CustomerId": "CustomerId_3638",
                "CurrencyCode": "UGX",
                "CountryCode": "UG",
                "ProviderId": "ProviderId_6",
                "ProductId": "ProductId_3",
                "ProductCategory": "airtime",
                "ChannelId": "ChannelId_3",
                "Amount": 256,
                "Value": 500,
                "DaysSinceFirstTransaction": 30,
                "TotalTransactionAmount": 1000,
                "AvgTransactionAmount": 500,
                "TransactionCount": 2,
                "StdTransactionAmount": 0,
                "MinTransactionAmount": 256,
                "MaxTransactionAmount": 256,
                "NumUniqueProducts": 1,
                "NumUniqueChannels": 1
            }
        }

class CreditRiskResponse(BaseModel):
    is_high_risk: int
    risk_probability: float