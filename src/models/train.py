"""
Training pipeline: Prophet + XGBoost models for energy load forecasting.
Logs metrics and model artifacts to MLflow.
"""

import logging
import warnings

import mlflow

from src.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MODELS_DIR,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_N_ESTIMATORS,
)
from src.models.prophet import train_prophet
from src.models.xgboost import train_xgboost
from src.utils.data_loader import load_and_prepare_data
from src.utils.file_ops import save_pickle

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure models directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Main training pipeline.

    Trains Prophet and XGBoost models on energy data and logs results to MLflow.
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load data
    data = load_and_prepare_data()

    with mlflow.start_run():
        # Train Prophet
        prophet_result = train_prophet(data.dataframe, data.date_column, data.target_column)
        mlflow.log_metrics(
            {
                "prophet_mae": prophet_result.metrics.mae,
                "prophet_rmse": prophet_result.metrics.rmse,
            }
        )

        # Save Prophet model using pickle (newer versions don't have .save() method)
        prophet_path = MODELS_DIR / "prophet_model.pkl"
        save_pickle(prophet_result.model, prophet_path, "Prophet model")
        mlflow.log_artifact(str(prophet_path), artifact_path="models/prophet")

        # Train XGBoost
        xgb_result = train_xgboost(data.dataframe, data.date_column, data.target_column)
        mlflow.log_metrics(
            {
                "xgboost_mae": xgb_result.metrics.mae,
                "xgboost_rmse": xgb_result.metrics.rmse,
            }
        )
        mlflow.log_params(
            {
                "n_estimators": XGBOOST_N_ESTIMATORS,
                "max_depth": XGBOOST_MAX_DEPTH,
                "learning_rate": XGBOOST_LEARNING_RATE,
            }
        )

        # Save XGBoost model
        xgb_path = MODELS_DIR / "xgb_model.pkl"
        save_pickle(xgb_result.model, xgb_path, "XGBoost model")
        mlflow.log_artifact(str(xgb_path), artifact_path="models/xgboost")

        # Save feature scaler
        scaler_path = MODELS_DIR / "xgb_scaler.pkl"
        save_pickle(xgb_result.scaler, scaler_path, "XGBoost feature scaler")
        mlflow.log_artifact(str(scaler_path), artifact_path="models/xgboost")

        # Save feature names
        feature_names_path = MODELS_DIR / "feature_names.pkl"
        save_pickle(xgb_result.feature_names, feature_names_path, "Feature names")
        mlflow.log_artifact(str(feature_names_path), artifact_path="models/xgboost")

        logger.info(f"Models saved to {MODELS_DIR}")
        logger.info(f"MLflow run ID: {mlflow.active_run().info.run_id}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
