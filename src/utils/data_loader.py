"""
Data loading and preparation utilities.
"""

import logging
from typing import NamedTuple

import pandas as pd

from src.constants import APPLIANCES_CSV, DATE_COLUMN, DEFAULT_START_DATE, FREQUENCY, TARGET_COLUMN

logger = logging.getLogger(__name__)


class DataLoadResult(NamedTuple):
    """Container for loaded and prepared data with column names."""

    dataframe: pd.DataFrame
    date_column: str
    target_column: str


def load_and_prepare_data() -> DataLoadResult:
    """
    Load appliances energy data and prepare for training.

    Returns:
        DataLoadResult with dataframe, date_column_name, and target_column_name

    Raises:
        FileNotFoundError: If appliances' data file does not exist
    """
    if not APPLIANCES_CSV.exists():
        msg = (
            f"Dataset not found at {APPLIANCES_CSV}. "
            f"Run 'python data/download_data.py' first to download real UCI data."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading data from {APPLIANCES_CSV}")
    df = pd.read_csv(APPLIANCES_CSV)

    # Ensure date column exists; create if missing
    if DATE_COLUMN not in df.columns:
        logger.warning(f"Date column '{DATE_COLUMN}' not found. " f"Creating synthetic dates.")
        df[DATE_COLUMN] = pd.date_range(
            start=DEFAULT_START_DATE,
            periods=len(df),
            freq=FREQUENCY,
        )

    # Standardize date column
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.dropna(subset=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} records from {df[DATE_COLUMN].min()} " f"to {df[DATE_COLUMN].max()}")

    return DataLoadResult(
        dataframe=df,
        date_column=DATE_COLUMN,
        target_column=TARGET_COLUMN,
    )
