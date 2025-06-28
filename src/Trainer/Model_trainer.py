# src/Trainer/Model_trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
import joblib


def load_data(filepath):
    """Load processed dataset and drop irrelevant columns"""
    df = pd.read_csv(filepath, low_memory=False)

    # Define target
    target = 'is_high_risk'

    # Auto-detect numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_features:
        numeric_features.remove(target)

    # Manually define categorical features (only those that exist in data)
    possible_categorical = ['ProductCategory']
    categorical_features = [col for col in possible_categorical if col in df.columns]

    all_features = numeric_features + categorical_features

    print("📊 Loaded features:", all_features)

    X = df[all_features]
    y = df[target].astype(int)

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def build_pipeline(model_name, numeric_features, categorical_features):
    """Build preprocessing + model pipeline based on available features"""
    if len(numeric_features) > 0:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    else:
        numeric_transformer = 'drop'

    if len(categorical_features) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
    else:
        categorical_transformer = 'drop'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='drop')

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    return full_pipeline


def train_model(model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a model using hyperparameter tuning"""
    print(f"\n🤖 Training {model_name}...")

    # Detect feature types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Build model pipeline
    full_pipeline = build_pipeline(model_name, numeric_features, categorical_features)

    if model_name == "random_forest":
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
    elif model_name == "logistic_regression":
        param_grid = {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__class_weight': [None, 'balanced']
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)

    try:
        grid_search.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        raise

    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    pred_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, pred_proba)
    }

    print(f"\nMetrics for {model_name}:\n", metrics)

    # Log to MLflow
    mlflow.set_tracking_uri("mlruns")
    with mlflow.start_run():
        mlflow.set_tag("model", model_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, model_name)

    # Save locally
    joblib.dump(best_model, f"models/{model_name}_best.pkl")

    return best_model, metrics


if __name__ == '__main__':
    DATA_PATH = 'Data/processed_data_with_risk.csv'

    print("🔄 Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    print("\n🤖 Training Random Forest...")
    rf_model, rf_metrics = train_model("random_forest", X_train, y_train, X_test, y_test)

    print("\n🤖 Training Logistic Regression...")
    lr_model, lr_metrics = train_model("logistic_regression", X_train, y_train, X_test, y_test)

    print("\n🏆 Best Model Metrics:")
    if rf_metrics['roc_auc'] > lr_metrics['roc_auc']:
        print(rf_metrics)
    else:
        print(lr_metrics)