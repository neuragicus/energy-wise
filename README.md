# Energy-Wise: Energy Load Forecasting with LLM Explanations

**A local machine learning system combining energy forecasting models with LLM-powered natural-language explanations.**

## What It Does

- **Forecasts** energy consumption up to 168 hours ahead using XGBoost and Prophet models
- **Explains** energy patterns using a LangChain agent with local Ollama LLM integration
- **Tracks** all training runs and metrics with MLflow
- **Serves** predictions via a FastAPI REST API (local development only)

### Key Features
- **Accurate predictions**: XGBoost achieves 30.82 kW MAE (10.6x better than baseline)
- **31 features**: Uses temperature, humidity, pressure, wind, lighting, and historical patterns
- **Local-first design**: Runs entirely on your machine (no cloud dependencies)
- **Explainable**: LangChain REACT agent provides natural-language explanations via Ollama
- **Experiment tracking**: MLflow integration for model versioning and metrics
- ️**Development stage**: Not yet containerized or production-ready

### Planned Architecture
More details on architecture in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Quick Start

### Prerequisites
- Python 3.12+
- Poetry
- **Ollama** (for LLM explanations) - [Download here](https://ollama.ai/)

### Installation & Training (10 minutes)

```bash
# 0. Start Ollama (required for LLM explanations)
#    In a separate terminal, run:
ollama serve

# In another terminal, download a model or use a cloud one via ollama
ollama pull <your favorite-model>

# 1. Clone and install
git clone https://github.com/neuragicus/energy-wise-lite.git
cd energy-wise-lite
poetry install --no-dev --no-root

# 2. Download data
python data/download_data.py
# This will download the UCI Appliances Energy Prediction dataset data/appliances_energy.csv
# 3. Train models
# This will create model artifacts in the models/ directory

python -m src.models.train
# Takes approximately 3-5 minutes depending on your machine
```

**Expected output:**
# Models are dumped into models/ directory
```
MLflow run ID: 553addfadfadgfadgagfadf
INFO - Training complete!

```

---

## Running the System

### Start the API Server
```bash
uvicorn src.api.app:app --reload
```

API available at: `http://127.0.0.1:8000`

### View API Documentation
Open `http://127.0.0.1:8000/docs` in your browser (interactive Swagger UI)

### View Training Metrics (MLflow Dashboard)
```bash
mlflow ui
```

Dashboard available at: `http://127.0.0.1:5000`

---

## API Endpoints

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Get 24-Hour Forecast
```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon": 24, "use_xgboost": true}'
```

**Response:**
```json
{
  "forecast": [374.83, 365.21, 352.10, ...],
  "timestamps": ["2026-01-17T10:00:00", "2026-01-17T11:00:00", ...],
  "model_used": "XGBoost",
  "horizon": 24
}
```

### Get Energy Explanation
```bash
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"question": "Why did load spike at 14:00?", "forecast_value": 65.5}'
```

**Response:**
```json
{
  "question": "Why did load spike at 14:00?",
  "explanation": "The energy load spike at 14:00 is typically driven by peak occupancy and HVAC demand during afternoon hours...",
  "timestamp": "2026-01-17T10:00:00"
}
```

---

## Project Structure
```
energy-wise/
├── src/
│   ├── api/
│   │   ├── app.py                  # FastAPI server with forecast & explain endpoints
│   │   ├── service.py              # Business logic layer
│   │   └── dtos.py                 # Request/response data models
│   ├── constants.py                # Configuration (features, hyperparameters)
│   ├── models/
│   │   ├── train.py                # Main training pipeline
│   │   ├── xgboost.py              # XGBoost trainer (31 features, feature scaling)
│   │   ├── prophet.py              # Prophet baseline model
│   │   └── metrics.py              # Model performance metrics
│   ├── llm/
│   │   └── energy_agent.py         # LangChain REACT agent for explanations
│   └── utils/
│       ├── data_loader.py          # Data loading and preprocessing
│       ├── file_ops.py             # File operations (model persistence)
│       └── model_loader.py         # Model artifact loading
├── data/
│   ├── download_data.py            # UCI dataset downloader
│   └── appliances_energy.csv       # Dataset (19,735 hourly records - available after download)
├── models/                         # Trained model artifacts
│   ├── xgb_model.pkl
│   ├── xgb_scaler.pkl
│   ├── prophet_model.pkl
│   └── feature_names.pkl
├── docs/
│   └── ARCHITECTURE.md             # Detailed system architecture documentation
├── mlruns/                         # MLflow experiment tracking data
└── pyproject.toml                  # Project metadata and dependencies
```

---

## Training Details

### Data
- **Source**: UCI Appliances Energy Prediction Dataset
- **Records**: 19,735 hourly observations (Jan-May 2016)
- **Features**: Appliance power, lights, temperatures, humidity, pressure, wind, visibility

### Models

#### XGBoost (Recommended)
- **Features**: 31 total
  - 25 sensor features (temperature, humidity, pressure, wind, etc.)
  - 3 temporal features (hour, day_of_week, month)
  - 3 lag features (1h, 24h, 7d patterns)
- **Accuracy**: MAE = 30.82 kW, RMSE = 67.29 kW
- **Training time**: ~1 second
- **Inference time**: ~100ms per prediction

#### Prophet (Baseline)
- **Type**: Time series forecasting with seasonality
- **Accuracy**: MAE = 327.81 kW, RMSE = 371.87 kW
- **Training time**: ~5 seconds
- **Use for**: Comparison and baseline

### Feature Scaling
Both models use `StandardScaler` to normalize features (mean=0, std=1) for stable learning across different feature ranges.

---

## Configuration

Edit `src/constants.py` to customize:

```python
# Model hyperparameters
XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 6
XGBOOST_LEARNING_RATE = 0.1

# Validation split
VALIDATION_HOURS = 30 * 24  # Last 30 days for validation

# Features
LAG_FEATURES = [1, 24, 168]  # Lag 1h, 24h, 7d
TEMPORAL_FEATURES = ["hour", "day_of_week", "month"]
```

---

## LLM Integration

### Current: Local Ollama (Required)
```bash
# Install Ollama: https://ollama.ai/
ollama serve

# The API will automatically connect to Ollama running on localhost:11434
# Make sure you are using a model that you actually have downloaded or can access,
# Set it up in OLLAMA_MODEL = "gpt-oss:120b-cloud" (src/llm/energy_agent.py)

```

Ollama must be running before starting the API server. The `/explain` endpoint will fail if Ollama is not available.

### Future: OpenAI API (Planned)
AI external service API (like OpenAI API) integration. To enable it when available:
```bash
# Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Restart the API server to load the key
uvicorn src.api.app:app --reload
```

### Fallback: MockLLMAgent
If needed, a `MockLLMAgent` with template-based responses can be used for testing without Ollama or OpenAI API.

---

## Performance Metrics

| Metric | Value                                 |
|--------|---------------------------------------|
| **XGBoost MAE** | 30.82 kW (Best for now)               |
| **Prophet MAE** | 327.81 kW (Baseline)                  |
| **Improvement** | 10.6x better than baseline            |
| **Features Used** | 31                                    |
| **Training Set** | 18,847 samples (96%)                  |
| **Validation Set** | 720 samples (4%)                      |
| **Model Size** | 0.37 MB (XGBoost) + 1.64 MB (Prophet) |

---

## Deployment

### Local Development (Current)
```bash
# Ensure Ollama is running in a separate terminal
ollama serve

# In another terminal:
python -m src.models.train          # Train models
uvicorn src.api.app:app --reload    # Start API
mlflow ui                           # View metrics
```

### Docker Container (Future)
Not yet containerized. Future work includes:
- Multi-stage Dockerfile for optimized image size
- Docker Compose for API + MLflow orchestration
- Container registry integration

When available:
```bash
docker build -f docker/Dockerfile -t energy-wise .
docker run -p 8000:8000 energy-wise

# Or just use docker-compose up when docker-compose.yml is ready, for local testing
```

### Google Cloud Run (Future)
Cloud deployment (no sense without containerization and a database with updated data)

---

## Troubleshooting

### Models Not Found
```bash
# Download the files
python -m data/download_data.py
# Re-train the models
python -m src.models.train
```

### Import Errors
```bash
# Reinstall dependencies
poetry install --no-cache
```

### API Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use a different port
uvicorn src.api:app --port 8001
```

### MLflow Dashboard Not Showing
```bash
# Ensure mlruns/ directory exists
ls -la mlruns/

# Start MLflow UI
mlflow ui
```

---

## Technologies

- **Machine Learning**: XGBoost, Prophet, Scikit-learn (future)
- **Data Processing**: Pandas, NumPy, PySpark (future)
- **API**: FastAPI, Uvicorn, Pydantic
- **LLM Integration**: LangChain, OpenAI (or local Ollama)
- **Experiment Tracking**: MLflow
- **Data Source**: UCI Machine Learning Repository

---

## License

Apache License 2.0

---

## Contributing

Contributions are always welcome, but I haven't setup rules yet - feel free to open issues or PRs.


## Areas for improvement/development:

### Authentication & Security
- API key authentication for `/forecast` and `/explain` endpoints
- Rate limiting and request throttling
- Prompt injection protection for LLM inputs

### Testing & Quality
- Unit tests for data loading, feature engineering, and model training
- Integration tests for API endpoints
- End-to-end tests for the full pipeline

### Data & Model Pipeline
- Live data integration from energy databases or IoT sensors
- Automated model retraining when significant new data becomes available
- Feature store for scalable feature management
- Distributed data processing (Spark ETL + Parquet) for production-scale datasets

### LLM & Explanations
- OpenAI API integration (alternative to local Ollama)
- Prompt optimization for better explanations

### Deployment & Infrastructure
- Dockerfile and Docker Compose for local development
- Kubernetes deployment manifests
- Google Cloud Run deployment configuration
- CI/CD pipeline with GitHub Actions

### Monitoring & Observability
- Request logging and performance metrics
- Model drift detection
- Alert thresholds for forecast accuracy degradation

---

## References

- **Dataset**: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction+data+set)
- **XGBoost**: [Documentation](https://xgboost.readthedocs.io/)
- **Prophet**: [Facebook's Time Series Library](https://facebook.github.io/prophet/)
- **LangChain**: [Framework for LLM Applications](https://www.langchain.com/)
- **FastAPI**: [Modern API Framework](https://fastapi.tiangolo.com/)

---

**Last Updated**: January 18, 2026
**Status**: 🚧 Development/Local Testing Only (Not Production-Ready)
**Author**: Neuragicus
