import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.api.main import app

client = TestClient(app)

def test_predict():
    payload = {
        "TransactionId": "TransactionId_12345",
        "BatchId": "BatchId_98765",
        "AccountId": "AccountId_456",
        "SubscriptionId": "SubscriptionId_789",
        "CustomerId": "CustomerId_012",
        "CurrencyCode": "UGX",
        "CountryCode": "UG",
        "ProviderId": "ProviderId_4",
        "ProductId": "ProductId_6",
        "ProductCategory": "financial_services",
        "ChannelId": "ChannelId_2",
        "Amount": 256,
        "Value": 5000,
        "DaysSinceFirstTransaction": 30,
        "TotalTransactionAmount": 10000,
        "AvgTransactionAmount": 2500,
        "TransactionCount": 4,
        "StdTransactionAmount": 1000,
        "MinTransactionAmount": 100,
        "MaxTransactionAmount": 5000,
        "NumUniqueProducts": 2,
        "NumUniqueChannels": 1
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "is_high_risk" in response.json()
    assert "risk_probability" in response.json()