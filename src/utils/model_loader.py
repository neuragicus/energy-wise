"""
Model loading utilities for the Energy-Wise API.

Provides consolidated functions to load XGBoost, Prophet, and feature models
with consistent error handling and logging.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

from prophet import Prophet

logger = logging.getLogger(__name__)


def load_model_from_pickle(
    filepath: Path,
    model_name: str,
    log_level: str = "info",
) -> Any | None:
    """
    Load a model from a pickle file with standardized error handling.

    Args:
        filepath: Path to the pickle file
        model_name: Human-readable name of the model (for logging)
        log_level: Logging level for success/failure ("info", "warning", "error")

    Returns:
        Loaded model object, or None if loading fails
    """
    if not filepath.exists():
        logger.debug(f"{model_name} file not found at {filepath}")
        return None

    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded {model_name}")
        return model
    except Exception as e:
        log_func = getattr(logger, log_level, logger.warning)
        log_func(f"Failed to load {model_name}: {e}")
        return None


def load_xgboost_model(models_dir: Path) -> tuple[Any | None, Any | None, list[str] | None]:
    """
    Load XGBoost model, scaler, and feature names.

    Args:
        models_dir: Directory containing model files

    Returns:
        Tuple of (xgb_model, xgb_scaler, feature_names)
        Any element may be None if loading fails
    """
    xgb_model = load_model_from_pickle(
        models_dir / "xgb_model.pkl",
        "XGBoost model",
        log_level="error",
    )

    xgb_scaler = load_model_from_pickle(
        models_dir / "xgb_scaler.pkl",
        "XGBoost feature scaler",
        log_level="warning",
    )

    feature_names = load_model_from_pickle(
        models_dir / "feature_names.pkl",
        "Feature names",
        log_level="error",
    )

    return xgb_model, xgb_scaler, feature_names


def load_prophet_model(models_dir: Path) -> Prophet | None:
    """
    Load Prophet model.

    Args:
        models_dir: Directory containing model files

    Returns:
        Loaded Prophet model, or None if loading fails
    """
    return load_model_from_pickle(
        models_dir / "prophet_model.pkl",
        "Prophet model",
        log_level="warning",
    )
