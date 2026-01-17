"""
Data Transfer Objects (DTOs) for the Energy-Wise Lite API.

Defines request and response models for all API endpoints.
"""

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request model for forecast endpoint."""

    horizon: int = Field(default=24, ge=1, le=168, description="Hours to forecast (1-168)")
    use_xgboost: bool = Field(default=True, description="Use XGBoost (True) or Prophet (False)")


class ForecastResponse(BaseModel):
    """Response model for forecast endpoint."""

    forecast: list[float]
    timestamps: list[str]
    model_used: str
    horizon: int


class ExplainRequest(BaseModel):
    """Request model for explain endpoint."""

    question: str = Field(description="Question about energy consumption")
    forecast_value: float = Field(default=62.5, description="Forecast value in kW")


class ExplainResponse(BaseModel):
    """Response model for explain endpoint."""

    question: str
    explanation: str
    timestamp: str
