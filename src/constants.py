"""
Configuration constants for energy-wise forecasting pipeline.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Dataset configuration
APPLIANCES_CSV = DATA_DIR / "appliances_energy.csv"
DATE_COLUMN = "date"
TARGET_COLUMN = "Appliances"

# Training hyperparameters
VALIDATION_DAYS = 30
VALIDATION_HOURS = VALIDATION_DAYS * 24

# XGBoost hyperparameters
XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1
XGBOOST_RANDOM_STATE = 42

# Feature engineering
LAG_FEATURES = [1, 24, 168]  # 1h, 24h, 7 days
TEMPORAL_FEATURES = ["hour", "day_of_week", "month"]

# Available sensor features from the dataset
# Temperature sensors in different zones (T1-T9)
# Relative humidity sensors in different zones (RH_1-RH_9)
# Outdoor and building properties
SENSOR_FEATURES = [
    "lights",  # Light usage
    "T1",
    "RH_1",  # Zone 1
    "T2",
    "RH_2",  # Zone 2
    "T3",
    "RH_3",  # Zone 3
    "T4",
    "RH_4",  # Zone 4
    "T5",
    "RH_5",  # Zone 5
    "T6",
    "RH_6",  # Zone 6
    "T7",
    "RH_7",  # Zone 7
    "T8",
    "RH_8",  # Zone 8
    "T9",
    "RH_9",  # Zone 9
    "T_out",  # Outdoor temperature
    "Press_mm_hg",  # Atmospheric pressure
    "RH_out",  # Outdoor relative humidity
    "Windspeed",  # Wind speed
    "Visibility",  # Visibility
    "Tdewpoint",  # Dew point temperature
]

# Prophet configuration
PROPHET_YEARLY_SEASONALITY = True
PROPHET_DAILY_SEASONALITY = True
PROPHET_INTERVAL_WIDTH = 0.95

# MLflow configuration
MLFLOW_EXPERIMENT_NAME = "energy-wise-forecasting"

# Time series frequency
FREQUENCY = "h"  # Hourly
DEFAULT_START_DATE = "2023-01-01"

# Model names
MODEL_XGBOOST = "XGBoost"
MODEL_PROPHET = "Prophet"
