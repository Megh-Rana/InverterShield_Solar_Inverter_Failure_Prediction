# ☀️ InverterShield — AI-Driven Solar Inverter Failure Prediction

> Predictive maintenance platform for solar inverters using XGBoost, SHAP explainability, and Generative AI insights.

## Architecture

```
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│   React Frontend │──────▶│   FastAPI Backend │──────▶│    ML Models     │
│   (Port 5173)    │  API  │   (Port 8000)    │       │  XGBoost + SHAP  │
│                  │       │                  │       │  Isolation Forest │
│  • Dashboard     │       │  • /predict      │       └──────────────────┘
│  • Predictions   │       │  • /dashboard    │                │
│  • Model Metrics │       │  • /chat         │       ┌────────▼─────────┐
│  • AI Copilot    │       │  • /health       │       │   GenAI Layer    │
└──────────────────┘       └──────────────────┘       │  Gemini / Ollama │
                                                      └──────────────────┘
```

## Model Performance

| Model                      | Metric         | Score |
|----------------------------|----------------|-------|
| Binary (Failure/No Fail)   | CV Mean F1     | 0.918 |
| Binary                     | CV Mean AUC    | 0.901 |
| Binary                     | CV Precision   | 0.924 |
| Binary                     | Holdout F1     | 0.908 |
| Multi-Class (3-way)        | CV F1 (macro)  | 0.789 |
| Anomaly Detection          | Isolation Forest | ✅    |

**Top SHAP Features:** `month`, `inv_temp_7d_mean`, `meter_kwh_import`, `inv_temp_24h_mean`, `smu_string_mean_7d_std`

## Dataset

- **3 solar plants** · 6 CSV files · 32 inverters · ~6M raw rows
- **2-year span**: March 2024 — March 2026
- Aggregated to **hourly intervals → 509,128 rows × 140 features**

## Quick Start

### Prerequisites
- Python 3.13+, Node.js 18+
- ~8GB RAM for training; inference < 2GB

### Install & Run

```bash
# 1. Clone
git clone https://github.com/Megh-Rana/InverterShield_Solar_Inverter_Failure_Prediction.git
cd InverterShield_Solar_Inverter_Failure_Prediction

# 2. Train models (optional — pre-trained models included via LFS)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --dataset-dir dataset --output-dir models --save-processed

# 3. Start API server
PYTHONPATH=. python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 4. Start React frontend (separate terminal)
cd frontend && npm install && npm run dev
```

### Docker

```bash
docker-compose up --build
# API:      http://localhost:8000
# Frontend: http://localhost:5173
```

### GenAI Setup (Optional)

```bash
export GEMINI_API_KEY=your_key_here    # For Gemini
ollama pull llama3                      # For local Ollama
```

## API Endpoints

| Endpoint         | Method | Description                     |
|------------------|--------|---------------------------------|
| `/health`        | GET    | Health check + model status     |
| `/predict`       | POST   | Single inverter prediction      |
| `/predict/batch` | POST   | Batch predictions               |
| `/dashboard`     | GET    | Dashboard summary + SHAP        |
| `/chat`          | POST   | GenAI chat (Gemini/Ollama)      |
| `/model/info`    | GET    | Model metrics & feature count   |

## Project Structure

```
├── frontend/                 # React + Vite dashboard
│   └── src/
│       ├── pages/            # Dashboard, Prediction, Performance, Chat
│       ├── api.js            # FastAPI client
│       ├── App.jsx           # Layout + routing
│       └── index.css         # Design system
├── src/
│   ├── api/main.py           # FastAPI backend
│   ├── genai/llm.py          # Gemini/Ollama integration
│   ├── ml/trainer.py         # XGBoost + SHAP + Isolation Forest
│   └── data_processing/      # CSV loader + feature engineering
├── models/                   # Trained models (Git LFS)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## ML Pipeline

1. **Load** — Normalize heterogeneous CSV schemas → unified per-inverter rows
2. **Aggregate** — 6M rows → 509K hourly rows (mean continuous, max alarms)
3. **Engineer** — 140 features: rolling stats, alarm features, KPIs, time features
4. **Detect** — Isolation Forest anomaly detection → scores as features
5. **Validate** — 5-fold walk-forward cross-validation (TimeSeriesSplit)
6. **Train** — XGBoost binary + multi-class classifiers
7. **Explain** — SHAP TreeExplainer for feature importance

## Hackathon Checklist

| Requirement                  | Status | Implementation                     |
|------------------------------|--------|-------------------------------------|
| Failure prediction model     | ✅      | XGBoost binary + multi-class       |
| 7-10 day prediction window   | ✅      | 7-day forward-looking target       |
| Feature engineering          | ✅      | 140 features                        |
| Cross-validation             | ✅      | Walk-forward 5-fold                 |
| Explainability (SHAP/LIME)   | ✅      | SHAP TreeExplainer                  |
| Anomaly detection            | ✅      | Isolation Forest                    |
| GenAI summaries              | ✅      | Gemini 2.0 Flash + Ollama fallback  |
| REST API                     | ✅      | FastAPI (6 endpoints)               |
| Interactive dashboard        | ✅      | React + Recharts                    |
| Docker containerization      | ✅      | Dockerfile + docker-compose         |

## Tech Stack

**ML:** XGBoost · scikit-learn · SHAP · pandas · numpy  
**API:** FastAPI · Uvicorn  
**Frontend:** React · Vite · Recharts · Framer Motion · Lucide  
**GenAI:** Google Gemini 2.0 Flash · Ollama  
**Infra:** Docker · docker-compose · Git LFS
