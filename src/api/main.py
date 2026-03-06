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


def prepare_features(features: Dict[str, float]) -> pd.DataFrame:
    """Prepare a feature dictionary into a DataFrame matching model expectations."""
    feature_names = models.get("feature_names", [])
    row = {f: features.get(f, 0.0) for f in feature_names}
    df = pd.DataFrame([row])
    # Ensure column order matches training
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_names]


def get_shap_explanation(X: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    """Get top SHAP features for a single prediction."""
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


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(data: InverterData):
    """Predict failure risk for a single inverter reading."""
    if "binary" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    X = prepare_features(data.features)

    # Binary prediction (failure probability)
    fail_prob = float(models["binary"].predict_proba(X)[0, 1])

    # Multi-class prediction (risk level)
    risk_class = int(models["multiclass"].predict(X)[0])
    risk_level = RISK_LABELS.get(risk_class, "unknown")

    # Anomaly detection
    anomaly_score = float(models["anomaly"].decision_function(X.drop(columns=["anomaly_score", "is_anomaly"], errors="ignore"))[0])
    is_anomaly = bool(models["anomaly"].predict(X.drop(columns=["anomaly_score", "is_anomaly"], errors="ignore"))[0] == -1)

    # SHAP explanation
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
