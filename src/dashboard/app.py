"""
Streamlit Dashboard for Solar Inverter Failure Prediction Platform.

Features:
- Real-time prediction with feature input
- Model performance metrics display
- SHAP feature importance visualization
- GenAI chat copilot for operational insights
- Multi-inverter risk overview
"""

import os
import json
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Solar Inverter AI Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))

RISK_COLORS = {"no_risk": "#2ecc71", "degradation_risk": "#f39c12", "shutdown_risk": "#e74c3c"}
RISK_LABELS = {0: "No Risk", 1: "Degradation Risk", 2: "Shutdown Risk"}
RISK_EMOJI = {0: "✅", 1: "⚠️", 2: "🚨"}


# ─── Data Loading (cached) ───────────────────────────────────────────────────

@st.cache_data
def load_metrics():
    try:
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data
def load_processed_data():
    parquet_path = os.path.join(DATA_DIR, "featured_data.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    return None


@st.cache_resource
def load_models():
    try:
        binary = joblib.load(os.path.join(MODEL_DIR, "binary_model.joblib"))
        multi = joblib.load(os.path.join(MODEL_DIR, "multiclass_model.joblib"))
        anomaly = joblib.load(os.path.join(MODEL_DIR, "anomaly_model.joblib"))
        features = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
        return {"binary": binary, "multiclass": multi, "anomaly": anomaly, "features": features}
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None


# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stMetric > div { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 15px; border-radius: 10px; border: 1px solid #0f3460; }
    .risk-card { padding: 20px; border-radius: 12px; margin: 10px 0; }
    .risk-no { background: linear-gradient(135deg, #1b4332, #2d6a4f); border: 1px solid #40916c; }
    .risk-deg { background: linear-gradient(135deg, #7f4f24, #936639); border: 1px solid #b08968; }
    .risk-shut { background: linear-gradient(135deg, #6a040f, #9d0208); border: 1px solid #d00000; }
    .chat-msg { padding: 12px; border-radius: 8px; margin: 5px 0; }
    .chat-user { background: #1a1a2e; border-left: 3px solid #e94560; }
    .chat-ai { background: #16213e; border-left: 3px solid #0f3460; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%); }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/solar-panel.png", width=60)
    st.title("☀️ SolarGuard AI")
    st.caption("Predictive Maintenance Platform")

    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "🔮 Live Prediction",
        "📊 Model Performance",
        "💬 AI Copilot",
    ], label_visibility="collapsed")

    st.divider()
    st.caption("Built with XGBoost • SHAP • Gemini")


# ─── Page: Dashboard ─────────────────────────────────────────────────────────

def page_dashboard():
    st.title("☀️ Solar Inverter Health Dashboard")
    st.markdown("**Real-time overview of all monitored inverters across 3 plants**")

    metrics = load_metrics()
    df = load_processed_data()
    models_dict = load_models()

    if metrics is None:
        st.warning("Model metrics not found. Run the training pipeline first.")
        return

    # ─── KPI Cards ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    binary_agg = metrics.get("cv_binary", {}).get("aggregate", {})
    with col1:
        st.metric("Model F1 Score", f"{binary_agg.get('f1_mean', 0):.3f}")
    with col2:
        st.metric("Model Precision", f"{binary_agg.get('precision_mean', 0):.3f}")
    with col3:
        st.metric("Model Recall", f"{binary_agg.get('recall_mean', 0):.3f}")
    with col4:
        st.metric("Inverters Monitored", "32" if df is not None else "N/A")

    st.divider()

    if df is not None and models_dict is not None:
        st.subheader("🏭 Plant Overview — Live Risk Assessment")

        # Get latest reading per inverter and predict
        latest = df.sort_values("datetime").groupby("inverter_id").tail(1).copy()
        feature_names = models_dict["features"]

        # Prepare features for prediction
        available_features = [f for f in feature_names if f in latest.columns]
        X_latest = latest[available_features].fillna(0).replace([np.inf, -np.inf], 0)

        # Pad missing features
        for f in feature_names:
            if f not in X_latest.columns:
                X_latest[f] = 0.0
        X_latest = X_latest[feature_names]

        # Predict
        probs = models_dict["binary"].predict_proba(X_latest)[:, 1]
        risk_classes = models_dict["multiclass"].predict(X_latest)

        latest = latest.copy()
        latest["failure_prob"] = probs
        latest["risk_class"] = risk_classes
        latest["risk_label"] = latest["risk_class"].map(RISK_LABELS)
        latest["risk_emoji"] = latest["risk_class"].map(RISK_EMOJI)

        # Display risk distribution
        rcol1, rcol2 = st.columns([1, 2])

        with rcol1:
            risk_counts = latest["risk_class"].value_counts().reindex([0, 1, 2], fill_value=0)
            fig_pie = px.pie(
                values=risk_counts.values,
                names=[RISK_LABELS[i] for i in risk_counts.index],
                color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                title="Risk Distribution"
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=300
            )
            st.plotly_chart(fig_pie, width="stretch")

        with rcol2:
            fig_bar = px.bar(
                latest.sort_values("failure_prob", ascending=False).head(15),
                x="inverter_id", y="failure_prob",
                color="risk_label",
                color_discrete_map={
                    "No Risk": "#2ecc71", "Degradation Risk": "#f39c12", "Shutdown Risk": "#e74c3c"
                },
                title="Top 15 Inverters by Failure Probability"
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white", height=300, xaxis_tickangle=-45
            )
            st.plotly_chart(fig_bar, width="stretch")

        # Risk table
        st.subheader("📋 All Inverters — Risk Status")
        display_df = latest[["inverter_id", "plant_id", "failure_prob", "risk_label"]].copy()
        display_df["failure_prob"] = display_df["failure_prob"].apply(lambda x: f"{x:.2%}")
        display_df.columns = ["Inverter ID", "Plant", "Failure Probability", "Risk Level"]
        st.dataframe(display_df.sort_values("Failure Probability", ascending=False), width="stretch", hide_index=True)

    # SHAP Feature Importance
    st.divider()
    st.subheader("🔍 Top Risk Factors (SHAP Feature Importance)")
    shap_features = metrics.get("top_shap_features", [])
    if shap_features:
        fig_shap = px.bar(
            x=list(range(len(shap_features))),
            y=shap_features,
            orientation="h",
            title="Top 10 Features Driving Failure Predictions",
            labels={"x": "Relative Importance", "y": "Feature"}
        )
        fig_shap.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=400, showlegend=False,
            yaxis=dict(autorange="reversed")
        )
        fig_shap.update_traces(marker_color="#e94560")
        st.plotly_chart(fig_shap, width="stretch")


# ─── Page: Live Prediction ───────────────────────────────────────────────────

def page_prediction():
    st.title("🔮 Live Inverter Risk Prediction")
    st.markdown("Enter inverter telemetry data or select an inverter from stored data")

    models_dict = load_models()
    df = load_processed_data()

    if models_dict is None:
        st.error("Models not loaded. Run the training pipeline first.")
        return

    tab1, tab2 = st.tabs(["📂 Select from Data", "✏️ Manual Input"])

    with tab1:
        if df is not None:
            inverter_ids = sorted(df["inverter_id"].unique())
            selected_inv = st.selectbox("Select Inverter", inverter_ids)

            inv_data = df[df["inverter_id"] == selected_inv].sort_values("datetime")

            # Show latest N readings
            n_recent = st.slider("Number of recent readings", 1, 50, 10)
            recent = inv_data.tail(n_recent)

            feature_names = models_dict["features"]
            available = [f for f in feature_names if f in recent.columns]
            X = recent[available].fillna(0).replace([np.inf, -np.inf], 0)
            for f in feature_names:
                if f not in X.columns:
                    X[f] = 0.0
            X = X[feature_names]

            probs = models_dict["binary"].predict_proba(X)[:, 1]
            risk_classes = models_dict["multiclass"].predict(X)

            # Show trend
            trend_df = pd.DataFrame({
                "Reading #": range(len(probs)),
                "Failure Probability": probs,
                "Risk Class": [RISK_LABELS[r] for r in risk_classes]
            })

            fig = px.line(
                trend_df, x="Reading #", y="Failure Probability",
                title=f"Failure Probability Trend — {selected_inv}",
                markers=True
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="white"
            )
            st.plotly_chart(fig, width="stretch")

            # Latest prediction detail
            col1, col2, col3 = st.columns(3)
            latest_prob = probs[-1]
            latest_risk = risk_classes[-1]

            with col1:
                st.metric("Failure Probability", f"{latest_prob:.2%}")
            with col2:
                st.metric("Risk Level", f"{RISK_EMOJI[latest_risk]} {RISK_LABELS[latest_risk]}")
            with col3:
                anomaly_score = float(models_dict["anomaly"].decision_function(
                    X.iloc[[-1]].drop(columns=["anomaly_score", "is_anomaly"], errors="ignore")
                )[0])
                st.metric("Anomaly Score", f"{anomaly_score:.3f}")

        else:
            st.info("No processed data found. Run the pipeline first with `--save-processed`.")

    with tab2:
        st.markdown("Enter key inverter parameters:")

        col1, col2 = st.columns(2)
        with col1:
            inv_power = st.number_input("Inverter Power (W)", 0, 100000, 5000)
            inv_temp = st.number_input("Inverter Temp (°C)", -10, 100, 45)
            meter_freq = st.number_input("Grid Frequency (Hz)", 45.0, 55.0, 50.0, step=0.1)

        with col2:
            alarm_count = st.number_input("Alarm Count (7d)", 0, 100, 0)
            pf = st.number_input("Power Factor", 0.0, 1.0, 0.98, step=0.01)
            hour = st.slider("Hour of Day", 0, 23, 12)

        if st.button("🔮 Predict Risk", type="primary"):
            features = {
                "inv_power": inv_power,
                "inv_temp": inv_temp,
                "meter_freq": meter_freq,
                "alarm_count_7d": alarm_count,
                "meter_pf": pf,
                "hour": hour,
                "is_daytime": 1 if 6 <= hour <= 18 else 0,
                "freq_deviation": abs(meter_freq - 50.0),
                "pf_deviation": abs(1.0 - pf),
            }

            feature_names = models_dict["features"]
            row = {f: features.get(f, 0.0) for f in feature_names}
            X = pd.DataFrame([row])[feature_names]

            prob = float(models_dict["binary"].predict_proba(X)[0, 1])
            risk = int(models_dict["multiclass"].predict(X)[0])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Failure Probability", f"{prob:.2%}")
            with col2:
                st.metric("Risk Level", f"{RISK_EMOJI[risk]} {RISK_LABELS[risk]}")

            if risk == 0:
                st.success("System is operating normally. No action needed.")
            elif risk == 1:
                st.warning("Performance degradation detected. Schedule inspection within 7 days.")
            else:
                st.error("HIGH SHUTDOWN RISK! Immediate inspection required.")


# ─── Page: Model Performance ─────────────────────────────────────────────────

def page_model_performance():
    st.title("📊 Model Performance Metrics")

    metrics = load_metrics()
    if metrics is None:
        st.error("Metrics not found. Run the training pipeline first.")
        return

    st.subheader("Binary Classification (Failure vs. No Failure)")

    binary_cv = metrics.get("cv_binary", {})
    binary_agg = binary_cv.get("aggregate", {})
    holdout = metrics.get("holdout_binary", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CV Mean F1", f"{binary_agg.get('f1_mean', 0):.3f}", f"±{binary_agg.get('f1_std', 0):.3f}")
    with col2:
        st.metric("CV Mean AUC", f"{binary_agg.get('auc_mean', 0):.3f}", f"±{binary_agg.get('auc_std', 0):.3f}")
    with col3:
        st.metric("Holdout F1", f"{holdout.get('f1', 0):.3f}")
    with col4:
        st.metric("Holdout AUC", f"{holdout.get('auc', 0):.3f}")

    # Per-fold chart
    folds = binary_cv.get("per_fold", [])
    if folds:
        fold_df = pd.DataFrame(folds)
        fig = go.Figure()
        for metric_name in ["precision", "recall", "f1", "auc"]:
            if metric_name in fold_df.columns:
                fig.add_trace(go.Scatter(
                    x=fold_df["fold"], y=fold_df[metric_name],
                    name=metric_name.upper(), mode="lines+markers"
                ))
        fig.update_layout(
            title="Binary Classification — Per-Fold Metrics",
            xaxis_title="Fold", yaxis_title="Score",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Multi-Class Classification (No Risk / Degradation / Shutdown)")

    multi_cv = metrics.get("cv_multiclass", {})
    multi_agg = multi_cv.get("aggregate", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CV Mean F1 (macro)", f"{multi_agg.get('f1_macro_mean', 0):.3f}")
    with col2:
        st.metric("CV Mean Precision", f"{multi_agg.get('precision_macro_mean', 0):.3f}")
    with col3:
        st.metric("CV Mean Recall", f"{multi_agg.get('recall_macro_mean', 0):.3f}")

    multi_folds = multi_cv.get("per_fold", [])
    if multi_folds:
        mfold_df = pd.DataFrame(multi_folds)
        fig2 = go.Figure()
        for metric_name in ["precision_macro", "recall_macro", "f1_macro"]:
            if metric_name in mfold_df.columns:
                fig2.add_trace(go.Scatter(
                    x=mfold_df["fold"], y=mfold_df[metric_name],
                    name=metric_name.replace("_", " ").upper(), mode="lines+markers"
                ))
        fig2.update_layout(
            title="Multi-Class — Per-Fold Metrics",
            xaxis_title="Fold", yaxis_title="Score",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig2, width="stretch")


# ─── Page: AI Copilot ────────────────────────────────────────────────────────

def page_chat():
    st.title("💬 AI Operations Copilot")
    st.markdown("Ask questions about inverter health, risk factors, or maintenance recommendations")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])
            if msg.get("source"):
                st.caption(f"Source: {msg['source']}")

    # Chat input
    user_input = st.chat_input("Ask about inverter health, risk factors, or maintenance...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Try API first, then fallback to direct
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"message": user_input},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("response", "No response")
                source = data.get("source", "api")
            else:
                raise Exception("API error")
        except Exception:
            # Direct fallback
            from src.genai.llm import _rule_based_response
            response_text = _rule_based_response(user_input)
            source = "local-fallback"

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_text,
            "source": source
        })
        st.chat_message("assistant").markdown(response_text)
        st.caption(f"Source: {source}")

    # Quick action buttons
    st.divider()
    st.markdown("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Model Performance"):
            st.session_state.chat_history.append({"role": "user", "content": "What is the model performance?"})
            st.rerun()
    with col2:
        if st.button("🔍 Top Risk Factors"):
            st.session_state.chat_history.append({"role": "user", "content": "What are the most important features?"})
            st.rerun()
    with col3:
        if st.button("🔧 Maintenance Tips"):
            st.session_state.chat_history.append({"role": "user", "content": "What maintenance actions do you recommend?"})
            st.rerun()


# ─── Router ───────────────────────────────────────────────────────────────────

if page == "🏠 Dashboard":
    page_dashboard()
elif page == "🔮 Live Prediction":
    page_prediction()
elif page == "📊 Model Performance":
    page_model_performance()
elif page == "💬 AI Copilot":
    page_chat()
