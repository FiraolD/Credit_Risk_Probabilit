# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime


class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, group_col='CustomerId'):
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        grouped = df.groupby(self.group_col).agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AvgTransactionAmount=('Amount', 'mean'),
            TransactionCount=('TransactionId', 'count'),
            StdTransactionAmount=('Amount', 'std')
        ).reset_index()

        # Merge back to original data
        df = df.merge(grouped, on=self.group_col, how='left')
        return df


class TimeBasedFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, time_col='TransactionStartTime'):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df['TransactionHour'] = df[self.time_col].dt.hour
        df['TransactionDay'] = df[self.time_col].dt.day
        df['TransactionMonth'] = df[self.time_col].dt.month
        df['TransactionYear'] = df[self.time_col].dt.year
        return df


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=['ProviderId', 'ProductId', 'ChannelId']):
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        self.label_maps_ = {}
        for col in self.cat_cols:
            unique_vals = X[col].unique()
            label_map = {val: idx for idx, val in enumerate(unique_vals)}
            self.label_maps_[col] = label_map
        return self

    def transform(self, X):
        df = X.copy()
        for col in self.cat_cols:
            df[col + '_label'] = df[col].map(self.label_maps_.get(col, {}))
        return df.drop(columns=self.cat_cols)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.impute_values_ = {}

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if self.strategy == 'median':
            self.impute_values_ = X[numeric_cols].median().to_dict()
        elif self.strategy == 'mean':
            self.impute_values_ = X[numeric_cols].mean().to_dict()
        else:
            self.impute_values_ = X[numeric_cols].mode().iloc[0].to_dict()
        return self

    def transform(self, X):
        df = X.copy()
        for col, value in self.impute_values_.items():
            df[col].fillna(value, inplace=True)
        return df


class NumericalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_type='standard'):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self.scaler_type = scaler_type
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.numeric_cols_ = numeric_cols
        self.scaler.fit(X[numeric_cols])
        return self

    def transform(self, X):
        df = X.copy()
        scaled = self.scaler.transform(df[self.numeric_cols_])
        for i, col in enumerate(self.numeric_cols_):
            df[col] = scaled[:, i]
        return df


class WoEBinner(BaseEstimator, TransformerMixin):
    def __init__(self, target='FraudResult', cols_to_encode=None):
        self.target = target
        self.cols_to_encode = cols_to_encode or ['ProviderId', 'ProductId', 'ChannelId']
        self.woe_maps_ = {}

    def fit(self, X, y):
        from woe.encoding import WOEEncoder
        df = X.copy()
        for col in self.cols_to_encode:
            encoder = WOEEncoder(variable=col, target=self.target)
            encoder.fit(df[[col, self.target]])
            self.woe_maps_[col] = encoder
        return self

    def transform(self, X):
        df = X.copy()
        for col, encoder in self.woe_maps_.items():
            df[col + '_woe'] = encoder.transform(df[[col]].fillna('missing'))[col]
        return df