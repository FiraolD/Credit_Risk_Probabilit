# proxy_target_variable.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load and clean processed data"""
    columns = [
        'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory',
        'ChannelId', 'Amount', 'Value', 'TransactionStartTime', 'PricingStrategy', 'FraudResult'
    ]
    
    df = pd.read_csv(filepath, low_memory=False)
    
    # Convert CustomerId to string to avoid merge issues
    df['CustomerId'] = df['CustomerId'].astype(str)
    
    # Ensure TransactionStartTime is parsed as datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    return df


def calculate_rfm(df, snapshot_date=None):
    """
    Calculate RFM metrics per customer
    """
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm


def create_clusters(rfm, n_clusters=3):
    """
    Create customer clusters using KMeans on scaled RFM features
    """
    # Replace infinities and fill NA
    rfm[['Recency', 'Frequency', 'Monetary']] = rfm[['Recency', 'Frequency', 'Monetary']].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Cluster customers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm, kmeans, scaler


def identify_high_risk_cluster(rfm):
    """
    Identify which cluster represents high-risk customers
    """
    cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    cluster_means['RiskScore'] = (cluster_means['Recency'] / (cluster_means['Frequency'] + 1)) * cluster_means['Monetary']
    
    sorted_clusters = cluster_means.sort_values(by='RiskScore', ascending=False)
    high_risk_cluster = sorted_clusters.index[0]
    
    print(f"🚨 Identified high-risk cluster: {high_risk_cluster}")
    print("\nCluster Means:")
    print(sorted_clusters)
    
    return high_risk_cluster


def add_risk_label_to_data(df, rfm_labeled, high_risk_cluster):
    """
    Add is_high_risk label back into main dataset
    """
    # Create binary risk label
    rfm_labeled['is_high_risk'] = (rfm_labeled['Cluster'] == high_risk_cluster).astype(int)

    # Ensure consistent types before merging
    rfm_labeled['CustomerId'] = rfm_labeled['CustomerId'].astype(str)
    df['CustomerId'] = df['CustomerId'].astype(str)

    # Merge with main dataset
    df_with_risk = df.merge(rfm_labeled[['CustomerId', 'is_high_risk']], 
                        on='CustomerId', how='left')

    # Fill missing values and ensure column exists
    if 'is_high_risk' in df_with_risk.columns:
        df_with_risk['is_high_risk'] = df_with_risk['is_high_risk'].fillna(0).astype(int)
    else:
        print("❌ 'is_high_risk' column could not be added — fallback to default")
        df_with_risk['is_high_risk'] = 0
    
    return df_with_risk


def main():
    DATA_PATH = 'Data/processed_data.csv'
    OUTPUT_PATH = 'Data/processed_data_with_risk.csv'

    print("🔄 Loading data...")
    df = load_data(DATA_PATH)

    print("📊 Calculating RFM metrics...")
    rfm = calculate_rfm(df)

    print("🔄 Creating customer clusters...")
    rfm_labeled, _, _ = create_clusters(rfm, n_clusters=3)

    print("🚨 Identifying high-risk cluster...")
    high_risk_cluster = identify_high_risk_cluster(rfm_labeled)

    print("🔄 Merging risk label into main dataset...")
    df_with_risk = add_risk_label_to_data(df, rfm_labeled, high_risk_cluster)

    print("💾 Saving updated dataset...")
    df_with_risk.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Done! File saved at: {OUTPUT_PATH}")

    print("\nHigh-risk cluster:", high_risk_cluster)
    print("\nRisk distribution:")
    print(df_with_risk['is_high_risk'].value_counts(normalize=True))


if __name__ == '__main__':
    main()