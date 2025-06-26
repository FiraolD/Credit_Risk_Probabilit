# src/pipelines/transaction_pipeline.py

from sklearn.pipeline import Pipeline
from src.feature_engineering import (
    AggregateCustomerFeatures,
    TimeBasedFeatureExtractor,
    CategoricalEncoder,
    MissingValueImputer,
    NumericalScaler,
    WoEBinner
)

def build_transaction_pipeline(**kwargs):
    """
    Builds a complete feature engineering pipeline
    """
    pipe = Pipeline([
        ('aggregate_features', AggregateCustomerFeatures()),
        ('time_extractor', TimeBasedFeatureExtractor()),
        ('missing_imputation', MissingValueImputer(strategy='median')),
        ('categorical_encoder', CategoricalEncoder()),
        ('woe_encoder', WoEBinner(target='FraudResult')),
        ('numerical_scaler', NumericalScaler(scaler_type='minmax'))
    ])
    return pipe