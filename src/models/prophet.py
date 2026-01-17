"""
Prophet time-series forecasting model training.
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.constants import (
    FREQUENCY,
    PROPHET_DAILY_SEASONALITY,
    PROPHET_INTERVAL_WIDTH,
    PROPHET_YEARLY_SEASONALITY,
    VALIDATION_HOURS,
)
from src.models.metrics import ModelMetrics

logger = logging.getLogger(__name__)


class ProphetModel(NamedTuple):
    """Container for Prophet model and metrics."""

    model: Prophet
    metrics: ModelMetrics


def train_prophet(df: pd.DataFrame, date_col: str, target_col: str) -> ProphetModel:
    """
    Train the Prophet model for time-series forecasting.

    Args:
        df: DataFrame with datetime and target columns
        date_col: Name of the datetime column
        target_col: Name of the target variable column

    Returns:
        ProphetModel containing trained model and metrics
    """
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
    prophet_df["y"] = prophet_df["y"].astype(float)

    # Split data (last VALIDATION_HOURS for validation)
    split_idx = len(prophet_df) - VALIDATION_HOURS
    train_df = prophet_df.iloc[:split_idx]

    logger.info("Training Prophet model...")
    model = Prophet(
        yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
        daily_seasonality=PROPHET_DAILY_SEASONALITY,
        interval_width=PROPHET_INTERVAL_WIDTH,
    )
    model.fit(train_df)

    # Forecast on the validation set
    future = model.make_future_dataframe(periods=VALIDATION_HOURS, freq=FREQUENCY)
    forecast = model.predict(future)

    # Evaluate on validation set
    val_actual = prophet_df.iloc[split_idx:]["y"].values
    val_pred = forecast.iloc[split_idx : split_idx + len(val_actual)]["yhat"].values

    mae = mean_absolute_error(val_actual, val_pred)
    rmse = np.sqrt(mean_squared_error(val_actual, val_pred))

    logger.info(f"Prophet - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return ProphetModel(model=model, metrics=ModelMetrics(mae=mae, rmse=rmse))
