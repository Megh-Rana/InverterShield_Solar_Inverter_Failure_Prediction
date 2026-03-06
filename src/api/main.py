"""
FastAPI Backend for Solar Inverter Failure Prediction Platform.

Endpoints:
- POST /predict       - Single inverter prediction
- POST /predict/batch - Batch prediction for multiple inverters
- GET  /dashboard     - Dashboard summary data
- POST /chat          - GenAI chat (Gemini/Ollama)
- GET  /health        - Health check
- GET  /model/info    - Model metrics & SHAP features
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Solar Inverter Failure Prediction API",
    description="AI-driven predictive maintenance for solar inverters",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Models on Startup ──────────────────────────────────────────────────

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))

models = {}


@app.on_event("startup")
async def load_models():
    """Load trained ML models into memory on startup."""
    try:
        models["binary"] = joblib.load(os.path.join(MODEL_DIR, "binary_model.joblib"))
        models["multiclass"] = joblib.load(os.path.join(MODEL_DIR, "multiclass_model.joblib"))
        models["anomaly"] = joblib.load(os.path.join(MODEL_DIR, "anomaly_model.joblib"))
        models["feature_names"] = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))

        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            models["metrics"] = json.load(f)

        # Pre-create SHAP explainer for binary model
        if HAS_SHAP:
            models["explainer"] = shap.TreeExplainer(models["binary"])

        print(f"✅ All models loaded from {MODEL_DIR}")
        print(f"   Features: {len(models['feature_names'])}")
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
        print("   API will work but predictions will fail until models are loaded.")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class InverterData(BaseModel):
    """Input data for a single inverter reading."""
    features: Dict[str, float]
    inverter_id: Optional[str] = "unknown"


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    readings: List[InverterData]


class PredictionResponse(BaseModel):
    """Single prediction response."""
    inverter_id: str
    failure_probability: float
    risk_level: str  # "no_risk", "degradation_risk", "shutdown_risk"
    risk_class: int  # 0, 1, 2
    is_anomaly: bool
    anomaly_score: float
    top_risk_factors: List[Dict]
    recommendation: str


class ChatRequest(BaseModel):
    """Chat request for GenAI copilot."""
    message: str
    inverter_id: Optional[str] = None
    context: Optional[Dict] = None


class ChatResponse(BaseModel):
    """Chat response."""
    response: str
    source: str  # "gemini" or "ollama" or "fallback"


# ─── Helper Functions ─────────────────────────────────────────────────────────

RISK_LABELS = {0: "no_risk", 1: "degradation_risk", 2: "shutdown_risk"}

RISK_RECOMMENDATIONS = {
    "no_risk": "System operating normally. Continue routine monitoring.",
    "degradation_risk": "⚠️ Performance degradation detected. Schedule inspection within 7 days. Check PV string connections and clean panels.",
    "shutdown_risk": "🚨 High shutdown risk! Immediate inspection required. Check alarm codes, temperature readings, and grid connection stability.",
}


# Try to load healthy baseline if available
BASELINE = {}
try:
    with open("healthy_baseline.json", "r") as f:
        BASELINE = json.load(f)
except Exception:
    pass

def prepare_features(features_dict: dict) -> pd.DataFrame:
    """Build a 30-feature vector for the retrained lightweight ML model.
    
    The model uses 30 key telemetry features directly available from
    inverter/meter/sensor readings. No rolling statistics needed.
    
    Strategy:
    1. Start from healthy baseline (median of no-risk samples)
    2. Override with user-provided telemetry values
    3. Compute derived features (voltage_imbalance, pf_deviation, etc.)
    4. Auto-compute time features from current clock if not provided
    """
    from datetime import datetime
    
    feature_names = models.get("feature_names", [])
    
    # 1. Start with healthy baseline (30 features, all realistic non-zero values)
    row = dict(BASELINE)
    for f in feature_names:
        if f not in row:
            row[f] = 0.0
    
    # 2. Override with ALL user-provided features
    for key, value in features_dict.items():
        if key in feature_names:
            row[key] = float(value)
    
    # 3. Time features: auto-fill from current time if not provided
    now = datetime.now()
    if "hour" not in features_dict:
        row["hour"] = float(now.hour)
    if "month" not in features_dict:
        row["month"] = float(now.month)
    row["is_daytime"] = 1.0 if 6 <= row["hour"] <= 18 else 0.0
    
    # 4. Alarm features: sync hours_since_last_alarm with alarm_count
    if "alarm_count_7d" in features_dict:
        a = float(features_dict["alarm_count_7d"])
        if "alarm_count_24h" not in features_dict:
            row["alarm_count_24h"] = min(a, a / 7.0 * 1.0)
        if a > 0:
            row["hours_since_last_alarm"] = max(1.0, 168.0 / (a + 1))
        else:
            row["hours_since_last_alarm"] = 9999.0
    
    # 5. Compute derived KPI features from input values
    if "meter_v_r" in row and "meter_v_y" in row and "meter_v_b" in row:
        voltages = [row["meter_v_r"], row["meter_v_y"], row["meter_v_b"]]
        avg_v = np.mean(voltages) if np.mean(voltages) > 0 else 1.0
        row["voltage_imbalance"] = (max(voltages) - min(voltages)) / avg_v
    
    if "meter_pf" in row:
        row["pf_deviation"] = abs(1.0 - row["meter_pf"])
    
    if "meter_freq" in row:
        row["freq_deviation"] = abs(row["meter_freq"] - 50.0)
    
    # power_ratio_vs_24h: if inv_power is provided but this isn't, estimate it
    if "inv_power" in features_dict and "power_ratio_vs_24h" not in features_dict:
        baseline_power = BASELINE.get("inv_power", 1.0)
        row["power_ratio_vs_24h"] = float(features_dict["inv_power"]) / (baseline_power + 0.01)
    
    if "smu_total_strings" in row and row["smu_total_strings"] > 0:
        row["smu_zero_fraction"] = row.get("smu_num_zero", 0) / row["smu_total_strings"]
    
    # Build DataFrame in exact model column order
    df = pd.DataFrame([row])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_names]


def get_shap_explanation(X: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    """Get top SHAP features for a single prediction using real ML explainability."""
    if "explainer" not in models:
        return []

    try:
        sv = models["explainer"].shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        if len(sv.shape) == 1:
            sv = sv.reshape(1, -1)

        abs_vals = np.abs(sv[0])
        top_idx = np.argsort(abs_vals)[-top_n:][::-1]

        feature_names = models["feature_names"]
        factors = []
        for idx in top_idx:
            factors.append({
                "feature": feature_names[idx],
                "value": round(float(X.iloc[0, idx]), 4),
                "impact": round(float(sv[0][idx]), 4),
                "direction": "increases risk" if sv[0][idx] > 0 else "decreases risk"
            })
        return factors
    except Exception:
        return []


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": bool(models.get("binary")),
        "version": "1.0.0"
    }


@app.get("/model/info")
async def model_info():
    """Return model metrics and SHAP feature importances."""
    if "metrics" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "metrics": models["metrics"],
        "feature_count": len(models.get("feature_names", [])),
        "model_type": "XGBoost",
    }


@app.get("/model/features")
async def model_features():
    """Return all feature names with their baseline values for the frontend."""
    if "feature_names" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "features": models["feature_names"],
        "baseline": BASELINE,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: InverterData):
    """Predict failure risk for a single inverter reading using ML models.
    
    Accepts any subset of the 30 telemetry features. Missing features are
    filled from the healthy baseline. Uses:
    - XGBoost multiclass classifier for risk class + calibrated probability
    - Isolation Forest for anomaly detection
    - SHAP TreeExplainer for feature importance
    """
    if "multiclass" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    X = prepare_features(data.features)

    # 1. Multi-class prediction — pure ML (primary output)
    multi_proba = models["multiclass"].predict_proba(X)[0]
    risk_class = int(np.argmax(multi_proba))
    risk_level = RISK_LABELS.get(risk_class, "unknown")
    
    # Calibrated failure probability: 1 - P(no_risk)
    # This gives a score from 0 to 1 that properly reflects risk
    fail_prob = float(1.0 - multi_proba[0])

    # 3. Anomaly detection — pure ML
    try:
        X_anomaly = X.drop(columns=["anomaly_score", "is_anomaly"], errors="ignore")
        anomaly_score = float(models["anomaly"].decision_function(X_anomaly)[0])
        is_anomaly = bool(models["anomaly"].predict(X_anomaly)[0] == -1)
    except Exception:
        anomaly_score = 0.0
        is_anomaly = False

    # 4. SHAP explanation — pure ML explainability
    top_factors = get_shap_explanation(X)

    return PredictionResponse(
        inverter_id=data.inverter_id or "unknown",
        failure_probability=round(fail_prob, 4),
        risk_level=risk_level,
        risk_class=risk_class,
        is_anomaly=is_anomaly,
        anomaly_score=round(anomaly_score, 4),
        top_risk_factors=top_factors,
        recommendation=RISK_RECOMMENDATIONS.get(risk_level, "Monitor closely.")
    )


@app.post("/predict/batch")
async def predict_batch(data: BatchPredictRequest):
    """Batch predict for multiple inverter readings."""
    results = []
    for reading in data.readings:
        result = await predict_single(reading)
        results.append(result)
    return {"predictions": results}


@app.get("/dashboard")
async def dashboard_data():
    """Provide summary data for the dashboard."""
    if "metrics" not in models:
        raise HTTPException(status_code=503, detail="Model metrics not loaded")

    metrics = models["metrics"]
    shap_features = metrics.get("top_shap_features", [])

    return {
        "model_performance": {
            "binary_cv": metrics.get("cv_binary", {}).get("aggregate", {}),
            "multiclass_cv": metrics.get("cv_multiclass", {}).get("aggregate", {}),
            "holdout_binary": metrics.get("holdout_binary", {}),
        },
        "top_risk_factors": shap_features,
        "risk_labels": RISK_LABELS,
        "total_features": len(models.get("feature_names", [])),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """GenAI chat endpoint — uses Gemini or Ollama for operational insights."""
    try:
        from src.genai.llm import get_llm_response
        response, source = await get_llm_response(
            message=request.message,
            inverter_id=request.inverter_id,
            context=request.context
        )
        return ChatResponse(response=response, source=source)
    except ImportError:
        # Fallback: rule-based response
        return ChatResponse(
            response=_fallback_chat(request.message),
            source="fallback"
        )
    except Exception as e:
        return ChatResponse(
            response=f"I encountered an error processing your request: {str(e)}. Please try rephrasing your question.",
            source="error"
        )


def _fallback_chat(message: str) -> str:
    """Rule-based fallback when GenAI is unavailable."""
    msg_lower = message.lower()
    if "shap" in msg_lower or "important" in msg_lower or "feature" in msg_lower:
        features = models.get("metrics", {}).get("top_shap_features", [])
        return f"The top risk-driving features are: {', '.join(features[:5])}. These were identified using SHAP analysis on the XGBoost binary classifier."
    elif "accuracy" in msg_lower or "performance" in msg_lower or "metric" in msg_lower:
        m = models.get("metrics", {}).get("cv_binary", {}).get("aggregate", {})
        return f"Binary classifier CV performance — Mean F1: {m.get('f1_mean', 'N/A'):.3f}, Mean AUC: {m.get('auc_mean', 'N/A'):.3f}, Mean Precision: {m.get('precision_mean', 'N/A'):.3f}"
    elif "risk" in msg_lower or "level" in msg_lower:
        return "Risk levels: 0=No Risk (system healthy), 1=Degradation Risk (performance drop, inspect in 7 days), 2=Shutdown Risk (immediate action needed, check alarms)."
    else:
        return "I'm the Solar Inverter AI Assistant. Ask me about model performance, SHAP feature importance, risk levels, or provide inverter data for analysis."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
