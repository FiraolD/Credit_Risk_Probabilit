# pipeline.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

# Import custom modules
from src.Data_Processor import (
    MissingValueHandler,
    TimeFeatureExtractor,
    AggregateCustomerFeatures,
    RFMClusterCreator,
    CategoricalEncoder,
    FeatureScaler
)


def load_data(filepath):
    """
    Load and clean raw transaction data
    """
    # Define expected column names
    expected_columns = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
            'ChannelId', 'Amount', 'Value', 'TransactionStartTime', 'PricingStrategy', 'FraudResult'
    ]
    
    # Load CSV with no header
    df = pd.read_csv(filepath, header=None, low_memory=False)
    
    print("Columns detected:", len(df.columns))
    
    if len(df.columns) != len(expected_columns):
        raise ValueError(f"Expected {len(expected_columns)} columns, got {len(df.columns)}")
    
    df.columns = expected_columns
    
    # Convert numeric fields
    for col in ['Amount', 'Value']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamp
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], format='%Y-%m-%dT%H:%M:%SZ', errors='coerce')

    return df


def build_feature_pipeline():
    """
    Build and return the complete feature engineering pipeline
    """
    return Pipeline([
        ('missing_imputation', MissingValueHandler(strategy='median')),
        ('time_extractor', TimeFeatureExtractor()),
        ('aggregate_features', AggregateCustomerFeatures(group_col='CustomerId')),
        ('rfm_creator', RFMClusterCreator(n_clusters=4)),
        ('categorical_encoder', CategoricalEncoder(cat_cols=['ProviderId', 'ProductId', 'ChannelId'])),
        ('feature_scaler', FeatureScaler(scaler_type='standard'))
    ])


def main():
    RAW_DATA_PATH = 'Data/data.csv'
    PROCESSED_DATA_PATH = 'Data/processed_data.csv'
    
    print("🚀 Starting full pipeline execution...")
    
    print("🔄 Loading raw data...")
    df_raw = load_data(RAW_DATA_PATH)
    
    print("🔄 Building pipeline...")
    pipeline = build_feature_pipeline()
    
    print("🔄 Applying feature engineering...")
    try:
        df_processed = pipeline.fit_transform(df_raw)
    except Exception as e:
        print(f"❌ Error during feature engineering: {str(e)}")
        return
    
    print("💾 Saving processed data...")
    try:
        df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"✅ Processed data saved to {PROCESSED_DATA_PATH}")
    except Exception as e:
        print(f"❌ Failed to save data: {str(e)}")
        return
    


if __name__ == '__main__':
    main()