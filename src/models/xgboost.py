"""
XGBoost gradient boosting model training with feature engineering.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.constants import (
    LAG_FEATURES,
    SENSOR_FEATURES,
    TEMPORAL_FEATURES,
    VALIDATION_HOURS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_N_ESTIMATORS,
    XGBOOST_RANDOM_STATE,
)
from src.models.metrics import ModelMetrics

logger = logging.getLogger(__name__)


class XGBoostResult(NamedTuple):
    """Container for XGBoost model, training data, and metrics."""

    model: XGBRegressor
    feature_names: list[str]
    metrics: ModelMetrics
    scaler: StandardScaler


def train_xgboost(df: pd.DataFrame, date_col: str, target_col: str) -> XGBoostResult:
    """
    Train XGBoost model with engineered features.

    Uses:
    - Sensor features (temperature, humidity, pressure, wind, etc.)
    - Lag features (1h, 24h, 7 days)
    - Temporal features (hour, day_of_week, month)

    Applies StandardScaler to stabilize learning across different feature scales.

    Args:
        df: DataFrame with datetime and target columns
        date_col: Name of the datetime column
        target_col: Name of the target variable column

    Returns:
        XGBoostResult containing trained model, feature names, metrics, and scaler
    """
    logger.info("Training XGBoost model...")

    # Feature engineering
    df_features = df.copy()

    # Add temporal features
    df_features["hour"] = df_features[date_col].dt.hour
    df_features["day_of_week"] = df_features[date_col].dt.dayofweek
    df_features["month"] = df_features[date_col].dt.month

    # Add lag features
    for lag in LAG_FEATURES:
        df_features[f"lag_{lag}"] = df_features[target_col].shift(lag)

    # Drop rows with NaN values from lag features
    df_features = df_features.dropna()

    # Select features: sensor features + temporal + lag features
    sensor_features_available = [col for col in SENSOR_FEATURES if col in df_features.columns]
    if not sensor_features_available:
        logger.warning("No sensor features found in dataset. Using temporal and lag features only.")
    else:
        logger.info(f"Using {len(sensor_features_available)} sensor features: {sensor_features_available}")

    lag_feature_cols = [f"lag_{lag}" for lag in LAG_FEATURES]
    feature_cols = sensor_features_available + TEMPORAL_FEATURES + lag_feature_cols

    # Ensure all feature columns exist
    feature_cols = [c for c in feature_cols if c in df_features.columns]

    logger.info(
        f"Total features: {len(feature_cols)} "
        f"(sensor: {len(sensor_features_available)}, temporal: {len(TEMPORAL_FEATURES)}, lag: {len(lag_feature_cols)})"
    )

    X = df_features[feature_cols].astype(float)
    y = df_features[target_col].astype(float)

    # Split data (last VALIDATION_HOURS for validation)
    split_idx = len(X) - VALIDATION_HOURS
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    # Scale features for better training stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train XGBoost
    logger.info(
        f"XGBoost hyperparameters: "
        f"n_estimators={XGBOOST_N_ESTIMATORS}, "
        f"max_depth={XGBOOST_MAX_DEPTH}, "
        f"learning_rate={XGBOOST_LEARNING_RATE}"
    )

    model = XGBRegressor(
        n_estimators=XGBOOST_N_ESTIMATORS,
        max_depth=XGBOOST_MAX_DEPTH,
        learning_rate=XGBOOST_LEARNING_RATE,
        random_state=XGBOOST_RANDOM_STATE,
        verbosity=0,
        objective="reg:squarederror",
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    logger.info(f"XGBoost - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return XGBoostResult(
        model=model,
        feature_names=feature_cols,
        metrics=ModelMetrics(mae=mae, rmse=rmse),
        scaler=scaler,
    )
