# src/Data_Processor.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values using specified strategy"""
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.numeric_imputer_ = SimpleImputer(strategy=strategy)
        self.cat_imputer_ = SimpleImputer(strategy='most_frequent')
        
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) > 0:
            self.numeric_imputer_.fit(X[numeric_cols])
        
        if len(cat_cols) > 0:
            self.cat_imputer_.fit(X[cat_cols])
            
        return self
    
    def transform(self, X):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.numeric_imputer_.transform(df[numeric_cols])
        
        if len(cat_cols) > 0:
            df[cat_cols] = self.cat_imputer_.transform(df[cat_cols])
        
        return df


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract time-based features from TransactionStartTime"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'TransactionStartTime' in df.columns:
            # Extract date parts
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionYear'] = df['TransactionStartTime'].dt.year
            df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
            df['IsWeekend'] = (df['TransactionDayOfWeek'] >= 5).astype(int)
            
            # Time since first transaction per customer
            customer_first = df.groupby('CustomerId')['TransactionStartTime'].min().reset_index()
            customer_first.columns = ['CustomerId', 'FirstTransactionDate']
            df = df.merge(customer_first, on='CustomerId', how='left')
            df['DaysSinceFirstTransaction'] = (df['TransactionStartTime'] - df['FirstTransactionDate']).dt.days
            
        return df


class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    """Creates aggregate features based on customer transaction history"""
    
    def __init__(self, group_col='CustomerId'):
        self.group_col = group_col
        self.aggregated_features_ = None
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Create grouped features
        grouped = df.groupby(self.group_col).agg(
            TotalTransactionAmount=('Amount', 'sum'),
            AvgTransactionAmount=('Amount', 'mean'),
            TransactionCount=('TransactionId', 'count'),
            StdTransactionAmount=('Amount', 'std'),
            MinTransactionAmount=('Amount', 'min'),
            MaxTransactionAmount=('Amount', 'max'),
            NumUniqueProducts=('ProductId', 'nunique'),
            NumUniqueChannels=('ChannelId', 'nunique')
        ).reset_index()
        
        # Merge back to original data
        df = df.merge(grouped, on=self.group_col, how='left')
        
        return df

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Processes categorical features using one-hot encoding"""
    
    def __init__(self, cat_cols=['ProviderId', 'ProductId', 'ChannelId']):
        self.cat_cols = [col for col in cat_cols if col in ['ProviderId', 'ProductId', 'ChannelId']]
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
    def fit(self, X, y=None):
        self.encoder.fit(X[self.cat_cols])
        return self
    
    def transform(self, X):
        df = X.copy()
        encoded_features = self.encoder.transform(df[self.cat_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=self.encoder.get_feature_names_out(self.cat_cols))
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
        return df.drop(columns=self.cat_cols)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scales numerical features to standard range"""
    
    def __init__(self, exclude_cols=['is_high_risk'], scaler_type='standard'):
        self.exclude_cols = exclude_cols
        self.scaler_type = scaler_type
        self.scaler_ = StandardScaler() if scaler_type == 'standard' else None
        self.cols_to_scale_ = None
        
    def fit(self, X, y=None):
        cols_to_scale = [col for col in X.columns 
                        if col not in self.exclude_cols and 
                        np.issubdtype(X[col].dtype, np.number)]
        self.cols_to_scale_ = cols_to_scale
        
        if len(cols_to_scale) > 0:
            self.scaler_.fit(X[cols_to_scale])
        
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.cols_to_scale_ and len(self.cols_to_scale_) > 0:
            scaled_data = self.scaler_.transform(df[self.cols_to_scale_])
            df[self.cols_to_scale_] = scaled_data
        
        return df