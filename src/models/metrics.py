"""
Container for model performance metrics.
"""

from typing import NamedTuple


class ModelMetrics(NamedTuple):
    """Container for model performance metrics."""

    mae: float
    rmse: float
