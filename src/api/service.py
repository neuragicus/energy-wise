"""
Business logic service layer for Energy-Wise API.

Handles forecasting and explanation generation with proper error handling.
"""

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from prophet import Prophet

from src.constants import FREQUENCY, MODEL_PROPHET, MODEL_XGBOOST

logger = logging.getLogger(__name__)


class ForecastService:
    """Service for generating energy load forecasts."""

    def __init__(
        self,
        xgb_model: object | None = None,
        xgb_scaler: object | None = None,
        prophet_model: Prophet | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize the forecast service with trained models.

        Args:
            xgb_model: Trained XGBoost model
            xgb_scaler: Feature scaler for XGBoost
            prophet_model: Trained Prophet model
            feature_names: Feature names for XGBoost
        """
        self.xgb_model = xgb_model
        self.xgb_scaler = xgb_scaler
        self.prophet_model = prophet_model
        self.feature_names = feature_names

    def forecast(
        self,
        horizon: int,
        use_xgboost: bool = True,
    ) -> tuple[list[float], list[str], str]:
        """
        Generate energy load forecast for the next N hours.

        Args:
            horizon: Number of hours to forecast
            use_xgboost: Use XGBoost (True) or Prophet (False)

        Returns:
            Tuple of (forecast_values, timestamps, model_name)

        Raises:
            ValueError: If no models are available for the requested forecast type
        """
        # Check if requested model is available
        if use_xgboost:
            if not self.xgb_model or not self.feature_names:
                raise ValueError(
                    "XGBoost model or feature names not available. " "Run 'python -m src.train' first to train models."
                )
            forecast_values = self._xgboost_forecast(horizon)
            model_name = MODEL_XGBOOST
        else:
            if not self.prophet_model:
                raise ValueError("Prophet model not available. " "Run 'python -m src.train' first to train models.")
            forecast_values = self._prophet_forecast(horizon)
            model_name = MODEL_PROPHET

        now = datetime.now(tz=UTC)
        timestamps = [(now + timedelta(hours=i)).isoformat() for i in range(1, horizon + 1)]

        return forecast_values, timestamps, model_name

    def _xgboost_forecast(self, horizon: int) -> list[float]:
        """
        Generate forecast using trained XGBoost model.

        Args:
            horizon: Number of hours to forecast

        Returns:
            List of predicted values for each hour
        """
        if not self.feature_names or not self.xgb_model:
            raise ValueError("XGBoost model or feature names not initialized")

        now = datetime.now(tz=UTC)

        # Build all feature vectors at once
        all_features = []
        for i in range(horizon):
            future_time = now + timedelta(hours=i)
            features = dict.fromkeys(self.feature_names, 0.0)
            features["hour"] = future_time.hour
            features["day_of_week"] = future_time.weekday()
            features["month"] = future_time.month
            all_features.append(features)

        # Create DataFrame with all features in correct order
        X = pd.DataFrame(all_features)[self.feature_names]

        # Apply feature scaling if available
        if self.xgb_scaler:
            X_scaled = self.xgb_scaler.transform(X)  # type: ignore[attr-defined]
        else:
            X_scaled = X.values

        # Batch predict all at once
        predictions = self.xgb_model.predict(X_scaled)  # type: ignore[attr-defined]

        # Vectorized clipping to ensure non-negative values
        forecast = np.maximum(predictions, 0).tolist()

        return forecast

    def _prophet_forecast(self, horizon: int) -> list[float]:
        """
        Generate forecast using trained Prophet model.

        Args:
            horizon: Number of hours to forecast

        Returns:
            List of predicted values for each hour
        """
        if not self.prophet_model:
            raise ValueError("Prophet model not initialized")

        future = self.prophet_model.make_future_dataframe(periods=horizon, freq=FREQUENCY)
        forecast_df = self.prophet_model.predict(future)

        # Get last 'horizon' rows and ensure non-negative values
        forecast_values = forecast_df["yhat"].tail(horizon).values
        return np.maximum(forecast_values, 0).tolist()
