"""
FastAPI server with two endpoints:
- POST /forecast: Generate 24-hour energy load forecast
- POST /explain: Generate natural-language explanation for forecast spike
"""

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import AgentExecutor
from prophet import Prophet

from src.api.dtos import ExplainRequest, ExplainResponse, ForecastRequest, ForecastResponse
from src.api.service import ForecastService
from src.constants import MODELS_DIR
from src.llm.energy_agent import generate_explanation, setup_llm_agent
from src.utils.model_loader import load_prophet_model, load_xgboost_model

# Configure logging
logger = logging.getLogger(__name__)

# Global models and service (initialized on startup)
xgb_model: Annotated[object | None, "Trained XGBoost model"] = None
xgb_scaler: Annotated[object | None, "Feature scaler for XGBoost"] = None
prophet_model: Annotated[Prophet | None, "Trained Prophet model"] = None
feature_names: Annotated[list[str] | None, "Feature names for XGBoost"] = None
llm_agent: Annotated[AgentExecutor | None, "LLM agent for explanations"] = None
forecast_service: Annotated[ForecastService | None, "Forecast service instance"] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Load models on startup and clean up on shutdown.

    This uses the new FastAPI lifespan context manager pattern.
    """
    global xgb_model, xgb_scaler, prophet_model, feature_names, llm_agent, forecast_service

    # Startup: Load trained models
    logger.info("Loading trained models...")

    # Load XGBoost model, scaler, and feature names
    xgb_model, xgb_scaler, feature_names = load_xgboost_model(MODELS_DIR)

    # Load Prophet model
    prophet_model = load_prophet_model(MODELS_DIR)

    # Initialize forecast service
    forecast_service = ForecastService(
        xgb_model=xgb_model,
        xgb_scaler=xgb_scaler,
        prophet_model=prophet_model,
        feature_names=feature_names,
    )

    # Initialize LLM agent
    try:
        llm_agent = setup_llm_agent()
        logger.info("Initialized LLM agent with Ollama")
    except ConnectionError as e:
        logger.warning(f"Could not initialize LLM agent: {e}")
        logger.info("Ensure Ollama is running: ollama serve")

    if not any([xgb_model, prophet_model]):
        logger.warning("No models found. Run 'python -m src.train' first to train models.")

    # Yield control to the application
    yield

    # TODO Shutdown: Gracefully clean up resources or close all pending request
    logger.info("Shutting down application...")


# Setup FastAPI app with lifespan
app = FastAPI(
    title="Energy-Wise Lite API",
    description="Energy load forecasting with LLM explanations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        Status information and timestamp
    """
    return {
        "status": "ok",
        "models_loaded": bool(xgb_model or prophet_model),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest) -> ForecastResponse:
    """
    Generate energy load forecast for the next N hours.

    Args:
        request: ForecastRequest with horizon and model choice

    Returns:
        ForecastResponse with predictions and timestamps

    Raises:
        HTTPException: 503 if no models are available, 400 if model not found
    """
    if not forecast_service:
        raise HTTPException(
            status_code=503,
            detail="Forecast service not initialized. Run 'python -m src.train' first.",
        )

    try:
        forecast_values, timestamps, model_name = forecast_service.forecast(
            horizon=request.horizon,
            use_xgboost=request.use_xgboost,
        )

        return ForecastResponse(
            forecast=forecast_values,
            timestamps=timestamps,
            model_used=model_name,
            horizon=request.horizon,
        )

    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest) -> ExplainResponse:
    """
    Generate natural-language explanation for energy consumption forecast.

    Uses LangChain agent with SQL and RAG tools.

    Args:
        request: ExplainRequest with question and forecast value

    Returns:
        ExplainResponse with natural-language explanation

    Raises:
        HTTPException: If LLM agent is not initialized or explanation fails
    """
    if not llm_agent:
        raise HTTPException(
            status_code=503,
            detail="LLM agent not initialized.",
        )

    try:
        explanation = generate_explanation(
            llm_agent,
            request.forecast_value,
            request.question,
        )

        return ExplainResponse(
            question=request.question,
            explanation=explanation,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.get("/")
async def root() -> dict:
    """
    API documentation and available endpoints.

    Returns:
        Dictionary with service information and available endpoints
    """
    return {
        "service": "Energy-Wise API",
        "version": "0.1.0",
        "description": "Energy load forecasting with LLM explanations",
        "endpoints": {
            "GET /health": {
                "description": "Health check endpoint",
                "returns": "Status, models availability, and timestamp",
            },
            "GET /docs": {"description": "Interactive API documentation (Swagger UI)"},
            "POST /forecast": {
                "description": "Generate energy load forecast",
                "parameters": {
                    "horizon": "Number of hours to forecast (1-168)",
                    "use_xgboost": "Use XGBoost (true) or Prophet (false)",
                },
                "errors": {"400": "Invalid request parameters", "503": "Models not available or training required"},
            },
            "POST /explain": {
                "description": "Generate natural-language explanation for energy patterns",
                "parameters": {
                    "question": "Question about energy consumption",
                    "forecast_value": "Forecast value in kW",
                },
                "errors": {"503": "LLM agent not initialized", "500": "Explanation generation failed"},
            },
        },
        "examples": {
            "forecast": {
                "endpoint": "POST /forecast",
                "request": {"horizon": 24, "use_xgboost": True},
                "response": {
                    "forecast": [60.5, 61.2, 62.1],
                    "timestamps": ["2026-01-17T12:00:00+00:00"],
                    "model_used": "XGBoost",
                    "horizon": 24,
                },
            },
            "explain": {
                "endpoint": "POST /explain",
                "request": {"question": "Why did load spike at 14:00?", "forecast_value": 65.5},
                "response": {
                    "question": "Why did load spike at 14:00?",
                    "explanation": "Based on historical patterns...",
                    "timestamp": "2026-01-17T11:30:00+00:00",
                },
            },
        },
        "status": "ready" if forecast_service else "models_not_loaded",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
