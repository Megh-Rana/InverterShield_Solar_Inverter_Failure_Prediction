# Technical Requirements Document (TRD)
**Project**: AI-Driven Solar Inverter Failure Prediction & Intelligence Platform

---

## 1. Data Pipeline & Feature Engineering

**Dataset**: 3 plants × 2 inverters each. Time-series telemetry at ~5-min intervals. Columns include PV string currents/voltages/power, inverter alarm codes, operational states, meter readings (voltage, frequency, power factor), SMU string currents, ambient temperature, and cumulative kWh readings.

**Target Variable**: Binary label — `1` if the inverter experiences a shutdown (`op_state` indicating fault, or critical `alarm_code`) or sustained underperformance (power output drops below threshold during daylight hours) within the next 7–10 days window.

**Feature Engineering**:
| Category | Features |
|---|---|
| **Raw Telemetry** | PV currents, voltages, power; inverter temperature; ambient temp |
| **Rolling Statistics** | Mean, std, min, max over 1h, 6h, 24h, 7d windows for key signals |
| **Computed KPIs** | Efficiency ratio (AC out / DC in), performance ratio, capacity utilization |
| **Alarm Features** | Alarm frequency counts (last 24h, 7d), time since last alarm, alarm type encoding |
| **Time Features** | Hour of day, day of week, month (seasonality proxies) |
| **Degradation Indicators** | Rate of change of efficiency, voltage imbalance across strings |

---

## 2. ML Model Pipeline

### Minimum Requirements → Implementation

| Requirement | How We Implement It |
|---|---|
| Binary classification or risk score | **XGBoost binary classifier** outputting `predict_proba` as a continuous risk score (0–1) |
| Feature engineering from telemetry, KPIs, alarms | See §1 above — rolling stats, computed KPIs, alarm counts |
| Cross-validation + hold-out with P/R/F1/AUC | **TimeSeriesSplit** (walk-forward) CV + final hold-out test set. Report all 4 metrics |
| SHAP for top 5 features | **`shap.TreeExplainer`** on XGBoost. Display waterfall/bar plots for top 5 |

### Bonus Requirements → Implementation

| Requirement | How We Implement It |
|---|---|
| Time-series aware splits | Walk-forward validation using `TimeSeriesSplit` (already in minimum) |
| Anomaly detection layer | **Isolation Forest** on telemetry features as a complementary signal; anomaly score fed as an additional feature to XGBoost |
| Multi-class output | 3-class: **No Risk / Degradation Risk / Shutdown Risk** (based on severity of future event within window) |

### Stretch Goal
- LSTM/GRU sequence model as a secondary experiment if time permits.

---

## 3. Generative AI Layer

### Minimum Requirements → Implementation

| Requirement | How We Implement It |
|---|---|
| Automated narrative generation | Prompt template receives risk score + SHAP top features → **Gemini 2.5 Flash** generates plain-English summary + recommended actions |
| RAG-based operator Q&A | Store inverter metadata + recent telemetry snapshots in **ChromaDB**. Operator queries are embedded, relevant context retrieved, and passed to LLM for grounded answers |
| Prompt design documentation | Document **≥2 prompt iterations** with rationale for improvements in `docs/prompt_engineering.md` |

### Bonus Requirements → Implementation

| Requirement | How We Implement It |
|---|---|
| Agentic workflow | **LangChain Agent** that autonomously retrieves inverter data, runs risk assessment via `/predict`, and drafts a maintenance ticket |
| Multi-turn conversational interface | Streamlit chat widget with **session-based context memory** (conversation history passed to LLM) |
| Hallucination guardrails | (1) System prompt strictly forbids fabrication, (2) all numerical claims are validated against the actual DataFrame before being returned to the user, (3) responses include source citations (inverter ID, timestamp) |

### LLM Provider Strategy
```
Priority 1: Gemini 2.5 Flash API (free tier, via google-genai SDK)
Priority 2: User-provided API key for any supported provider
Priority 3: Local Ollama model (llama3 / mistral) for offline/high-resource environments
```

---

## 4. Backend / API (FastAPI)

| Requirement | Endpoint / Implementation |
|---|---|
| Predict endpoint | `POST /predict` — accepts inverter telemetry JSON, returns `{ risk_score, risk_class, top_features, narrative }` |
| Health endpoint | `GET /health` — returns `{ status: "ok", model_loaded: true }` |
| Input validation | **Pydantic** models with informative error messages for missing/invalid fields |
| Containerization | `Dockerfile` + `docker-compose.yml` (FastAPI + Streamlit services) |

Additional endpoints:
- `POST /chat` — accepts operator question + context, returns grounded LLM response
- `GET /inverters` — lists all inverters with latest risk scores

---

## 5. Frontend Dashboard (Streamlit)

| Requirement | Implementation |
|---|---|
| Per-inverter risk scores | Color-coded cards/table with risk scores (green/yellow/red) |
| Trend visualizations | Plotly time-series charts: power output, temperature, efficiency over time |
| GenAI narrative summary | Expandable section per inverter showing the LLM-generated explanation |
| Responsive design (bonus) | Streamlit's native responsive layout + `st.columns` for adaptive grid |

Additional features:
- **Chat interface** (`st.chat_input`) for operator Q&A
- **Plant/Inverter selector** sidebar for navigation

---

## 6. Code Quality

| Requirement | Implementation |
|---|---|
| Modular, well-commented code | Clear project structure (see §7) |
| `requirements.txt` with pinned versions | Generated via `pip freeze` after development |
| ≥3 unit tests | `pytest` tests for: (1) feature engineering pipeline, (2) `/predict` endpoint, (3) `/health` endpoint |

---

## 7. Project Structure

```
solar_inverter/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── data/                    # Raw + processed datasets
├── notebooks/               # EDA and experimentation
├── src/
│   ├── data_processing/     # Data loading, cleaning, feature engineering
│   ├── ml/                  # Model training, evaluation, SHAP
│   ├── genai/               # LLM integration, prompts, RAG
│   ├── api/                 # FastAPI app, routes, schemas
│   └── dashboard/           # Streamlit app
├── models/                  # Saved model artifacts (.joblib)
├── tests/                   # Unit tests
└── docs/                    # Prompt engineering docs, architecture diagram
```

---

## 8. Verification Plan

- **ML**: Walk-forward CV → report P/R/F1/AUC on hold-out set
- **API**: `pytest` against `/predict`, `/health`, `/chat`
- **E2E**: `docker-compose up` → verify Streamlit dashboard communicates with FastAPI
- **GenAI**: Manual review of generated narratives for accuracy against source data
