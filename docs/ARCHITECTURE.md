# Energy-Wise - Architecture & Design

**Current Status**: Development-stage, local machine implementation (not containerized or production-ready)

## 🏗️ System Architecture (At least, how is planned to work)

**Current Implementation** (Single Machine, Local Ollama)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER (out of scope in this project)  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Web Browser │  │  Mobile App  │  │  Monitoring Dashboard    │   │
│  └──────┬───────┘  └──────┬───────┘  └────────── ┬──────────────┘   │
│         │                 │                      │                  │
└─────────┼─────────────────┼──────────────────────┼──────────────────┘
          │                 │                      │
          └─────────────────┼──────────────────────┘
                            │
         ┌──────────────────▼────────────────────────┐
         │    FastAPI Server (Port 8000)             │
         │  ┌─────────────────────────────────────┐  │
         │  │  GET  /                             │  │ ← API Docs
         │  │  GET  /health                       │  │ ← Health Check
         │  │  GET  /docs (Swagger)               │  │ ← Interactive
         │  │  POST /forecast  {horizon: 24}      │  │ ← Predictions
         │  │  POST /explain   {question, value}  │  │ ← LLM Assist
         │  └─────────────────────────────────────┘  │
         └──────────┬────────────────────────────────┘
                    │
         ┌──────────┴────────────────────────────-┐
         │                                        │
    ┌────▼──────────────┐               ┌─────────▼─────────┐
    │  Forecasting Engines              │  LLM Agent        │
    │  ┌────────────────┐               │  ┌──────────────┐ │
    │  │ Prophet Model  │               │  │ LangChain    │ │
    │  │ (Fast, Simple) │               │  │ ┌─ SQL Tool  │ │
    │  └────────────────┘               │  │ └─ RAG Tool  │ │
    │  ┌────────────────┐               │  └──────────────┘ │
    │  │ XGBoost Model  │               │                   │
    │  │ (Accurate)     │               │  LLM Backend:     │
    │  └────────────────┘               │  ├─ Ollama        │
    │                                   │  ├─ WhateverAI API│
    │                                   │  └─ MockLLM (?)   │
    └────────┬───────────────────────── ┘───────────────────┘
             │
    ┌────────▼────────────┐
    │  Data & Models      │
    │  ┌────────────────┐ │
    │  │ xgb_model.pkl  │ │
    │  │ prophet_model/ │ │
    │  │ feature_names  │ │
    │  └────────────────┘ │
    └─────────────────────┘

    └────────────────────────────────────────────────────────┘
                        MLflow Tracking Server
                     (Metrics, Models, Versioning)
                     http://127.0.0.1:5000
```

---

## 📊 Data Flow Diagram

```
┌────────────────────┐
│  UCI Datasets      │
│  • Appliances      │
│  • Energy Eff.     │
└─────────┬──────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  data/download_data.py               │
  │  • Fetch from UCI ML Repository      │
  │  • Requires internet connection      │
  │  • Save as CSV                       │
  └─────────┬────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │  Pandas Data Processing              │
  │  • Date parsing & sorting            │
  │  • Train/validation split (30d hold) │
  │  • Feature engineering               │
  └─────────┬────────────────────────────┘
            │
      ┌─────┴──────┐
      │            │
      ▼            ▼
  Prophet      XGBoost
  Training     Training
      │             │
      └─────┬───────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │  MLflow Experiment Tracking          │
  │  • Log MAE, RMSE metrics             │
  │  • Register models                   │
  │  • Version artifacts                 │
  │  • UI at http://127.0.0.1:5000       │
  └─────────┬────────────────────────────┘
            │
            ▼
  ┌──────────────────────────────────────┐
  │  models/ Directory                   │
  │  • xgb_model.pkl                     │
  │  • prophet_model/                    │
  │  • feature_names.pkl                 │
  └──────────────────────────────────────┘
```

---

## 🔄 Inference Pipeline

```
┌─────────────────────────────────────┐
│  CLIENT REQUEST                     │
│  POST /forecast {horizon: 24}       │
└────────────────┬────────────────────┘
                 │
                 ▼
      ┌──────────────────────────┐
      │  FastAPI Request Handler │
      │  • Validate input        │
      │  • Create feature vector │
      │  • Check model loaded    │
      └────────────┬─────────────┘
                   │
           ┌───────┴────────┐
           │                │
           ▼                ▼
       Prophet          XGBoost
       Forecast         Forecast
       (24 values)      (24 values)
           │                │
           └────────┬───────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │  POST-PROCESSING         │
         │  • Ensure non-negative   │
         │  • Create timestamps     │
         │  • JSON format response  │
         └──────────┬───────────────┘
                    │
                    ▼
    ┌──────────────────────────────────┐
    │  JSON Response                   │
    │  {                               │
    │    "forecast": [63.2, 62.8, ...],│
    │    "timestamps": [...],          │
    │    "model_used": "XGBoost",      │
    │    "horizon": 24                 │
    │  }                               │
    └──────────────────────────────────┘
```

---

## 🤖 LLM Agent Flow

```
┌────────────────────────────────────────┐
│  CLIENT REQUEST                        │
│  POST /explain                         │
│  {question, forecast_value}            │
└─────────────────┬──────────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │  LangChain Agent Initialization     │
    │  • Select LLM (OpenAI/Ollama/Mock)  │
    │  • Prepare tools                    │
    │  • Set up callbacks                 │
    └────────────────┬────────────────────┘
                     │
                     ▼
      ┌──────────────────────────────────┐
      │  Agent Decision Making           │
      │  (REACT Pattern)                 │
      │  Reason → Act → Observe          │
      └────────────────┬─────────────────┘
                       │
          ┌────────────┴────────────────┐
          │                             │
          ▼                             ▼
    ┌──────────────┐          ┌─────────────────┐
    │ SQL Tool     │          │ RAG Tool        │
    │ • Query hist.│          │ • Retrieve tips │
    │ • Get data   │          │ • Match facts   │
    └──────────────┘          └─────────────────┘
          │                             │
          └────────────────┬────────────┘
                           │
                           ▼
           ┌────────────────────────────────┐
           │  LLM Processing                │
           │  • Combine context             │
           │  • Generate response           │
           │  • Temperature: 0.3 (focused)  │
           └────────────────┬───────────────┘
                            │
                            ▼
          ┌──────────────────────────────────┐
          │  JSON Response                   │
          │  {                               │
          │    "question": "...",            │
          │    "explanation": "...",         │
          │    "timestamp": "..."            │
          │  }                               │
          └──────────────────────────────────┘
```

---

## 🐳 Docker Build Pipeline

```
Dockerfile (Multi-stage)
│
├─ STAGE 1: Builder
│  ├─ Base: python:3.12-slim
│  ├─ Install build tools
│  ├─ Copy pyproject.toml
│  ├─ Poetry install
│  ├─ python data/download_data.py
│  ├─ python src/train.py
│  └─ Output: trained models, venv
│
└─ STAGE 2: Runtime
   ├─ Base: python:3.12-slim (clean)
   ├─ Install runtime libs only
   ├─ Copy venv from stage 1
   ├─ Copy models from stage 1
   ├─ Expose port 8000
   ├─ Health check
   └─ CMD uvicorn src.api:app

Result: ~500-700 MB image
(vs ~2GB if single stage)
```

---

## 🔄 Deployment Flow

### CURRENT STATUS: Single Machine Development
```
┌──────────────────────────────────────────┐
│  LOCAL DEVELOPMENT (✅ Implemented)      │
│  • Poetry install                        │
│  • python data/download_data.py          │
│  • python -m src.models.train            │
│  • uvicorn src.api.app:app               │
│  • Requires: Ollama running locally      │
│  • Single machine, single process        │
└──────────────────────────────────────────┘
```

### FUTURE STAGES (Planned but Not Yet Implemented)

```
STAGE 2: Docker Containers (🔜 Planned)
    ├─ Multi-stage Dockerfile
    ├─ Docker Compose for API + MLflow
    └─ Local container orchestration

STAGE 3: Cloud Deployment (🔜 Planned)
    ├─ Google Cloud Run
    ├─ Kubernetes manifests
    └─ CI/CD pipeline (GitHub Actions)

STAGE 4: Scalable Infrastructure (🔜 Future)
    ├─ Load Balancer
    ├─ API Replicas (K8s)
    ├─ Model Server (Ray Serve)
    ├─ Message Queue (RabbitMQ)
    ├─ Cache Layer (Redis)
    └─ Database (PostgreSQL)
```

---

## 🔄 ML Pipeline Details

### Current Data Pipeline (✅ Implemented)

```
Static UCI Dataset (Offline)
    │
    ├─ data/download_data.py
    │  └─ Fetch once from UCI ML Repository
    │     (19,735 hourly records, Jan-May 2016)
    │
    └─ src/utils/data_loader.py
       ├─ Load CSV into Pandas DataFrame
       ├─ Parse dates and sort chronologically
       ├─ 30-day validation split
       └─ Feature engineering (temporal + lag)

Training Pipeline (src/models/train.py)
    │
    ├─ Load & prepare data
    ├─ Train Prophet model
    ├─ Train XGBoost model
    ├─ Log metrics & artifacts to MLflow
    └─ Save to models/ directory

Inference Pipeline (src/api/service.py)
    │
    ├─ Load pre-trained models
    ├─ Build feature vectors for forecast period
    ├─ Scale features with StandardScaler
    ├─ Generate predictions
    └─ Return as JSON response
```

### Prophet Model Details

```
Time Series Decomposition
    │
    ├─ Trend Component
    │  └─ Piecewise linear regression with changepoints
    │
    ├─ Seasonality Components
    │  ├─ Yearly (12-month cycle)
    │  ├─ Weekly (7-day cycle)
    │  └─ Daily (24-hour cycle)
    │
    └─ Holiday Effects (Future Enhancement)
       └─ Special dates with custom seasonality

Training Characteristics:
    • Training time: ~5 seconds on validation set
    • Fast inference: suitable for real-time applications
    • Handles missing data automatically
    • Interpretable components: easy to understand patterns
    • Assumes historical patterns will repeat

Performance on Energy Dataset:
    • MAE: 327.81 kW (baseline)
    • RMSE: 371.87 kW
    • Best for: Understanding trend and seasonality
    • Limitation: Less accurate than XGBoost
```

### XGBoost Model Details

```
Feature Engineering (31 Total Features)
    │
    ├─ Temporal Features (3)
    │  ├─ Hour of day (0-23)
    │  ├─ Day of week (0-6)
    │  └─ Month (1-12)
    │
    ├─ Lag Features (3)
    │  ├─ lag_1: Previous hour consumption
    │  ├─ lag_24: Previous day, same hour
    │  └─ lag_168: Previous week, same hour
    │
    └─ Sensor Features (25)
       ├─ Temperatures (T1-T9, T_out)
       ├─ Humidity (RH_1 to RH_9, RH_out)
       ├─ Atmospheric pressure (Press_mm_hg)
       ├─ Wind speed
       ├─ Visibility
       ├─ Dew point
       └─ Lighting consumption

Gradient Boosting Configuration:
    • Algorithm: XGBoost (eXtreme Gradient Boosting)
    • N estimators: 100 decision trees
    • Max depth: 6 (shallow trees, avoid overfitting)
    • Learning rate: 0.1 (conservative updates)
    • Objective: Regression (Mean Squared Error)
    • Validation: 30-day holdout set (720 samples, 4%)

Feature Scaling:
    • StandardScaler (mean=0, std=1)
    • Fit on training data only
    • Applied before both training and inference
    • Ensures stable gradient-based learning

Training Characteristics:
    • Training time: ~1 second
    • Inference time: ~100ms for 24-hour forecast
    • Handles non-linear relationships
    • Feature importance computed
    • Best for: High-accuracy predictions

Performance on Energy Dataset:
    • MAE: 30.82 kW (Excellent) ✅
    • RMSE: 67.29 kW
    • 10.6x better than Prophet baseline
    • Explains ~87% of variance
```

### Future Data Pipeline (🔜 Planned)

```
Live Energy Database Integration
    │
    ├─ Real-time IoT/sensor connections
    │  └─ Building energy management systems
    │
    ├─ Scheduled data polling
    │  └─ Fetch new data points hourly/daily
    │
    └─ src/utils/data_pipeline.py (Future)
       ├─ Database connection management
       ├─ Data validation & quality checks
       └─ Incremental dataset updates

Automated Model Retraining (🔜 Planned)
    │
    ├─ Trigger conditions:
    │  ├─ New data available (weekly/monthly)
    │  ├─ Model drift detection threshold exceeded
    │  └─ Performance degradation beyond acceptable
    │
    └─ Implementation options:
       ├─ Scheduled cron jobs
       ├─ Event-based triggers (message queue)
       └─ Manual retraining via API endpoint

Scalable ETL Pipeline (🔜 Planned)
    │
    ├─ For production-scale datasets:
    │  ├─ Apache Spark for distributed processing
    │  ├─ Parquet format for efficient storage
    │  ├─ Delta Lake for ACID transactions
    │  └─ Feature store (e.g., Tecton, Feast)
    │
    └─ Benefits:
       ├─ Handle millions of data points
       ├─ Incremental updates efficiency
       ├─ Multi-model feature sharing
       └─ Point-in-time correct training
```

---

## 🔐 Security Considerations

### Current Implementation

```
┌────────────────────────────────────────┐
│  API Security         │
│  • CORS enabled for local development  │
│  • Pydantic validation (input safety)  │
│  • Error handling (no stack leaks)     │
│  • Health check endpoint               │
│  • No authentication (local only)      │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  LLM Security                          │
│  • Local Ollama only (no API keys)     │
│  • No external API calls in current    │
│  • Fallback to MockLLMAgent            │
│  • Input validation via Pydantic       │
└────────────────────────────────────────┘

```

### Future Security Enhancements (🔜 Planned)

```
┌────────────────────────────────────────┐
│  API Authentication                    │
│  • API key authentication for endpoints│
│  • Bearer token support                │
│  • Rate limiting and throttling        │
│  • Request logging and audit trail     │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  LLM & Prompt Security                 │
│  • OpenAI (or other) API key management|
     (.env)                              │
│  • Prompt injection protection         │
│  • Output filtering for sensitive data │
│  • LLM response validation             │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│  Infrastructure Security               │
│  • TLS/HTTPS for API endpoints         │
│  • Secrets management (Vault/Sealed)   │
│  • Database connection encryption      │
│  • Container image scanning            │
└────────────────────────────────────────┘
```

---

## 📈 Scalability Path

### CURRENT IMPLEMENTATION (Single Machine)
```
STAGE 1: Local Single-Machine Setup
    │
    └─ Characteristics:
       • Single FastAPI server + Models on localhost
       • Ollama LLM running locally
       • No data storage (just a static dataset)
       • MLflow tracking on local filesystem
       • Handles: ~50-100 req/min (CPU-bound)
       • Best for: Development and testing

    Resources Used:
       • CPU: 2-4 cores
       • Memory: 2-4 GB
       • Disk: 500 MB - 1 GB
```

### FUTURE SCALING ROADMAP (Definitely unnecessary for now)

```
STAGE 2: Containerized Local Development (Planned)
    │
    └─ Characteristics:
       • Docker Compose with API + MLflow services
       • Network bridge between containers
       • Local development parity with production
       • Prerequisite for cloud deployment

    Benefits:
       • Consistent environments (dev/prod)
       • Easy dependency management
       • Reproducible setups


STAGE 3: Cloud-Ready Single Instance (Planned)
    │
    └─ Characteristics:
       • Google Cloud Run deployment
       • Serverless execution model
       • Auto-scaling based on demand
       • HTTPS out-of-the-box
       • Pay-per-request billing
       • Handles: ~1000 req/min (auto-scaled)


STAGE 4: Distributed Microservices (
    │
    ├─ API Service
    │  └─ Multiple replicas (Kubernetes)
    │
    ├─ Model Service
    │  └─ Dedicated model serving (TensorFlow Serving)
    │
    ├─ LLM Service
    │  └─ Async task queue (Celery + RabbitMQ)
    │
    ├─ Data Pipeline
    │  └─ Spark ETL + Parquet data lake
    │
    ├─ Database Layer
    │  └─ PostgreSQL for structured data
    │
    ├─ Cache Layer
    │  └─ Redis for predictions caching
    │
    └─ Monitoring & Observability
       ├─ Prometheus (metrics)
       ├─ Grafana (dashboards)
       └─ ELK Stack (logging)

    Handles: 10k+ req/min
    Production-grade workloads
```

---

## Design Philosophy (at least what I tried to follow)

**KISS (Keep It Simple Stupid)**
- All core code fits in ~550 lines
- No complex orchestration required for development
- Single-machine setup for learning and exploration
- Clear path to cloud always in mind

**DRY (Don't Repeat Yourself)**
- Reusable components (models, services, utilities)
- Configuration via constants.py
- MLflow for centralized experiment tracking

**YAGNI (You Aren't Gonna Need It)**
- Only essential features implemented
- No over-engineering for future scenarios

**Fail Fast, Recover Gracefully**
- Validate input early with Pydantic
- Try/except with informative error messages
- Health checks for service monitoring
- Graceful degradation when optional services unavailable

**Clear Development Roadmap**
- Current: Local machine learning and development
- Future: Containerized, cloud-deployable, scalable
- Well-documented transition path to production

---

**Last Updated**: January 17, 2026
**Project Status**: Development/Local Testing Only
