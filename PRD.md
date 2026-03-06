# Product Requirements Document (PRD)
**Project Title**: AI-Driven Solar Inverter Failure Prediction & Intelligence Platform

## 1. Objective
To build an AI-powered platform that predicts the likelihood of solar inverter shutdowns or significant underperformance within a 7-10 day window. The platform will augment these predictions with Generative AI to provide explainable insights, root-cause hypotheses, and actionable recommendations in natural language.

## 2. Target Audience
- **Solar Plant Operators**: Need early warnings to schedule maintenance and prevent energy loss.
- **Maintenance Engineers**: Need clear, actionable insights on what components to check based on predicted risks.

## 3. Core Features & Requirements

### 3.1. Predictive Visibility (ML Component)
- **Capability**: Predict the risk (binary or regression score) of an inverter failing or underperforming in the next 7-10 days.
- **Explainability**: Display the top 5 contributing factors for the prediction (using SHAP or LIME) so operators trust the model.
- **Inputs**: Raw telemetry data (voltages, currents, temperatures, power), computed KPIs, and historical alarm data.

### 3.2. Generative Insight Engine (GenAI Component)
- **Automated Narratives**: Translate the prediction risk score and SHAP feature importances into a plain-English summary explaining *why* the inverter is at risk and *what* actions to take.
- **Operator Q&A (RAG)**: A conversational interface allowing operators to ask questions (e.g., "Which inverters have elevated risk this week?") and receive grounded, data-backed answers without hallucinated data.

### 3.3. Software & Interface Requirements
- **Backend API**: REST API (e.g., FastAPI) to serve predictions and host the GenAI endpoints. Must include health checks and input validation.
- **Frontend Dashboard**: Minimal, responsive UI (Streamlit or React) visualizing risk scores, historical trends, and the natural language summaries per inverter.
- **Deployment**: The entire stack must be containerized using Docker and `docker-compose`.

## 4. Out of Scope (MVP)
- Automated control systems (the AI will not automatically shut down or restart inverters).
- General grid issue prediction (focus is strictly on inverter health and SMU/PV panel underperformance feeding into the inverter).

## 5. Success Metrics
- **ML Performance**: High Recall (minimizing false negatives/missed failures) and reasonable Precision, evaluated via cross-validation and hold-out sets. AUC and F1-score reported.
- **GenAI Quality**: Accuracy of the generated summaries matching the underlying tabular data (no hallucinations).
- **Usability**: Dashboard loads quickly and provides an intuitive overview of plant health.
