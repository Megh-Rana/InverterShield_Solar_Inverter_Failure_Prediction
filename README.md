# ☀️ SolarGuard AI — Solar Inverter Failure Prediction Platform

AI-driven predictive maintenance platform for solar inverters, built for hackathon submission. Uses XGBoost for failure prediction, SHAP for explainability, and Generative AI for operational insights.

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit   │────▶│  FastAPI      │────▶│  XGBoost    │
│  Dashboard   │     │  Backend API  │     │  Models     │
│  (Port 8501) │     │  (Port 8000)  │     │  + SHAP     │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  GenAI       │
                    │  Gemini /    │
                    │  Ollama      │
                    └──────────────┘
```

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Binary (Failure/No Failure) | CV Mean F1 | 0.918 |
| Binary | CV Mean Precision | 0.924 |
| Binary | CV Mean AUC | 0.901 |
| Multi-Class (No Risk/Degradation/Shutdown) | CV Mean F1 (macro) | 0.789 |
| Anomaly Detection | Isolation Forest | ✅ |

**Top SHAP Features:** month, inv_temp_7d_mean, meter_kwh_import, inv_temp_24h_mean, smu_string_mean_7d_std

## 📁 Dataset

- **3 solar plants**, 6 CSV files, ~6M raw rows
- **32 unique inverters** across varying schemas (1-12 inverters per data logger)
- **2-year span**: March 2024 — March 2026
- Heterogeneous schemas normalized into unified per-inverter format
- Aggregated to hourly intervals → **509,128 rows × 140 features**

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- ~8GB RAM (for model training; inference needs <2GB)

### Install & Run

```bash
# Clone and setup
cd solar_inverter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train models (takes ~24 min)
python run_pipeline.py --dataset-dir dataset --output-dir models --save-processed

# Run API server
PYTHONPATH=. python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Run Dashboard (separate terminal)
PYTHONPATH=. python -m streamlit run src/dashboard/app.py --server.port 8501
```

### Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

### GenAI Setup (Optional)

```bash
# For Gemini (recommended)
export GEMINI_API_KEY=your_key_here

# For Ollama (local, no API key needed)
ollama pull llama3
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single inverter prediction |
| `/predict/batch` | POST | Batch predictions |
| `/dashboard` | GET | Dashboard summary data |
| `/chat` | POST | GenAI chat (Gemini/Ollama) |
| `/model/info` | GET | Model metrics & SHAP features |

### Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "features": {
      "inv_power": 5000,
      "inv_temp": 55,
      "meter_freq": 49.8,
      "alarm_count_7d": 3
    },
    "inverter_id": "Plant_1_INV0"
  }'
```

## 📂 Project Structure

```
solar_inverter/
├── run_pipeline.py           # ML pipeline orchestrator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── dataset/                  # Raw CSV telemetry data
│   ├── Plant 1/             # 2 loggers, 23 inverters
│   ├── Plant 2/             # 2 loggers, 7 inverters
│   └── Plant 3/             # 2 loggers, 2 inverters
├── data/processed/           # Featured data (parquet)
├── models/                   # Trained model artifacts
│   ├── binary_model.joblib
│   ├── multiclass_model.joblib
│   ├── anomaly_model.joblib
│   ├── feature_names.joblib
│   └── metrics.json
├── src/
│   ├── data_processing/
│   │   ├── loader.py         # CSV loading + hourly aggregation
│   │   └── features.py       # Feature engineering
│   ├── ml/
│   │   └── trainer.py        # XGBoost + SHAP + Isolation Forest
│   ├── api/
│   │   └── main.py           # FastAPI backend
│   ├── genai/
│   │   └── llm.py            # Gemini/Ollama integration
│   └── dashboard/
│       └── app.py            # Streamlit frontend
└── docs/
    ├── PRD.md
    └── TRD.md
```

## 🧠 ML Pipeline

1. **Data Loading**: Normalize heterogeneous CSV schemas → unified per-inverter rows
2. **Hourly Aggregation**: 6M rows → 509K rows (mean for continuous, max for alarms)
3. **Feature Engineering**: 140 features including rolling stats, alarm features, KPIs
4. **Anomaly Detection**: Isolation Forest adds anomaly scores as features
5. **Walk-Forward CV**: 5-fold time-series aware cross-validation
6. **XGBoost Training**: Binary (failure/no failure) + Multi-class (no risk/degrade/shutdown)
7. **SHAP Analysis**: TreeExplainer for feature importance and per-prediction explanations

## 📋 Hackathon Requirements Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| ML model for failure prediction | ✅ | XGBoost binary + multi-class |
| 7-10 day prediction window | ✅ | 7-day forward-looking target |
| Feature engineering | ✅ | 140 features (rolling, alarm, KPI) |
| Cross-validation | ✅ | Walk-forward (TimeSeriesSplit) 5-fold |
| Model explainability (SHAP/LIME) | ✅ | SHAP TreeExplainer |
| Anomaly detection | ✅ | Isolation Forest |
| GenAI summaries | ✅ | Gemini 1.5 Flash + Ollama fallback |
| REST API | ✅ | FastAPI with 6 endpoints |
| Interactive dashboard | ✅ | Streamlit with 4 pages |
| Docker containerization | ✅ | Dockerfile + docker-compose |

## 🛠 Tech Stack

- **ML**: XGBoost, scikit-learn, SHAP, pandas
- **API**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **GenAI**: Google Gemini 1.5 Flash, Ollama (local)
- **Infra**: Docker, docker-compose
- **Data**: pyarrow (parquet), numpy
