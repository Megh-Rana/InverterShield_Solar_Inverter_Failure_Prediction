"""
Retrain a lightweight XGBoost model on key telemetry features.

This trains models that work with the ~20 key telemetry features 
an operator can actually provide (or that come from real-time telemetry).
The original 142-feature model relied on rolling window statistics that 
can't be computed from a single data point.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
from sklearn.ensemble import IsolationForest

# Key features available from inverter telemetry
KEY_FEATURES = [
    # Inverter
    "inv_power", "inv_temp", "inv_freq",
    "inv_v_ab", "inv_v_bc", "inv_v_ca",
    "inv_pv1_power", "inv_kwh_today",
    # Meter
    "meter_pf", "meter_freq",
    "meter_v_r", "meter_v_y", "meter_v_b",
    "meter_meter_active_power",
    # Environment
    "ambient_temp",
    # String monitoring
    "smu_string_mean", "smu_string_std", "smu_num_zero", "smu_total_strings",
    # Alarms
    "alarm_count_24h", "alarm_count_7d", "hours_since_last_alarm",
    # Time
    "hour", "month", "is_daytime",
    # Derived KPIs
    "voltage_imbalance", "pf_deviation", "freq_deviation",
    "power_ratio_vs_24h", "smu_zero_fraction",
]

print("=" * 60)
print("  Retraining Lightweight ML Models")
print("=" * 60)

# Load the processed data
DATA_DIR = "data"
processed_file = os.path.join(DATA_DIR, "processed", "featured_data.parquet")

if not os.path.exists(processed_file):
    print(f"❌ {processed_file} not found.")
    print("Please run the pipeline first: python run_pipeline.py")
    exit(1)

print("Loading processed data...")
df = pd.read_parquet(processed_file)
print(f"Shape: {df.shape}")

# Check available columns
available_features = [f for f in KEY_FEATURES if f in df.columns]
missing = [f for f in KEY_FEATURES if f not in df.columns]
print(f"\nAvailable key features: {len(available_features)}/{len(KEY_FEATURES)}")
if missing:
    print(f"Missing: {missing}")

# Check targets
if "target_binary" not in df.columns:
    print("❌ target_binary not in data. Re-running feature engineering.")
    exit(1)

# Build feature matrix — fill NaN with 0 (many meters weren't connected)
X = df[available_features].fillna(0).copy()
y_binary = df["target_binary"].copy()

# Handle multi-class target
if "target_multiclass" in df.columns:
    y_multi = df["target_multiclass"].fillna(0).copy()
else:
    y_multi = y_binary.copy()

# Only drop rows where the target is NaN
mask = y_binary.notna()
X = X[mask].reset_index(drop=True)
y_binary = y_binary[mask].astype(int).reset_index(drop=True)
y_multi = y_multi[mask].astype(int).reset_index(drop=True)

print(f"\nTraining data shape: {X.shape}")
print(f"Binary target distribution:\n{y_binary.value_counts()}")
print(f"Multi-class distribution:\n{y_multi.value_counts()}")

# ─── Train Binary Classifier ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Training Binary XGBoost Classifier")
print("=" * 60)

n_neg = (y_binary == 0).sum()  # no-risk (minority)
n_pos = (y_binary == 1).sum()  # failure (majority)
scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0  # < 1 = penalise majority

print(f"  Class balance: no_risk={n_neg}, failure={n_pos}, scale_pos_weight={scale_pos_weight:.3f}")

binary_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False,
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    binary_model, X, y_binary, cv=cv,
    scoring=["f1", "precision", "recall", "roc_auc"],
    return_train_score=False,
)

print(f"\n  5-Fold CV Results:")
print(f"  F1:        {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
print(f"  AUC:       {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")

# Train on full data
binary_model.fit(X, y_binary)

# ─── Train Multi-class Classifier ────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Training Multi-class XGBoost Classifier")
print("=" * 60)

# Compute sample weights to balance ALL 3 classes equally
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_multi)
print(f"  Multiclass distribution: {y_multi.value_counts().to_dict()}")

multi_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss",
    use_label_encoder=False,
    objective="multi:softprob",
    num_class=3,
)

cv_multi = cross_validate(
    multi_model, X, y_multi, cv=cv,
    scoring=["f1_weighted", "precision_weighted", "recall_weighted"],
    return_train_score=False,
)

print(f"\n  5-Fold CV Results:")
print(f"  F1 (weighted):        {cv_multi['test_f1_weighted'].mean():.4f} ± {cv_multi['test_f1_weighted'].std():.4f}")
print(f"  Precision (weighted): {cv_multi['test_precision_weighted'].mean():.4f} ± {cv_multi['test_precision_weighted'].std():.4f}")
print(f"  Recall (weighted):    {cv_multi['test_recall_weighted'].mean():.4f} ± {cv_multi['test_recall_weighted'].std():.4f}")

multi_model.fit(X, y_multi, sample_weight=sample_weights)

# ─── Train Anomaly Detector ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  Training Isolation Forest")
print("=" * 60)

anomaly_model = IsolationForest(
    n_estimators=100,
    contamination=0.15,
    random_state=42,
    n_jobs=-1,
)
anomaly_model.fit(X)
print("  Isolation Forest trained.")

# ─── Save Models ─────────────────────────────────────────────────────────────

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(binary_model, os.path.join(MODEL_DIR, "binary_model.joblib"))
joblib.dump(multi_model, os.path.join(MODEL_DIR, "multiclass_model.joblib"))
joblib.dump(anomaly_model, os.path.join(MODEL_DIR, "anomaly_model.joblib"))
joblib.dump(available_features, os.path.join(MODEL_DIR, "feature_names.joblib"))

# Compute healthy baseline from NO_RISK class
# Use the actual data distribution to find the "most healthy" values
healthy = df[df["target_binary"] == 0][available_features].fillna(0)
baseline = {}
for col in available_features:
    vals = healthy[col]
    baseline[col] = round(float(vals.median()), 6)

# Print baseline for verification
print(f"\n  Baseline (from no-risk median):")
for k, v in baseline.items():
    print(f"    {k}: {v}")

with open("healthy_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print(f"\n  Saved baseline with {len(baseline)} features")

# Save metrics
metrics = {
    "cv_binary": {
        "aggregate": {
            "f1_mean": float(cv_results["test_f1"].mean()),
            "f1_std": float(cv_results["test_f1"].std()),
            "precision_mean": float(cv_results["test_precision"].mean()),
            "precision_std": float(cv_results["test_precision"].std()),
            "recall_mean": float(cv_results["test_recall"].mean()),
            "recall_std": float(cv_results["test_recall"].std()),
            "auc_mean": float(cv_results["test_roc_auc"].mean()),
            "auc_std": float(cv_results["test_roc_auc"].std()),
        }
    },
    "cv_multiclass": {
        "aggregate": {
            "f1_weighted_mean": float(cv_multi["test_f1_weighted"].mean()),
            "precision_weighted_mean": float(cv_multi["test_precision_weighted"].mean()),
            "recall_weighted_mean": float(cv_multi["test_recall_weighted"].mean()),
        }
    },
    "feature_count": len(available_features),
    "training_samples": len(X),
    "top_shap_features": [],  # Will compute separately
}

# Compute SHAP feature importance
try:
    import shap
    explainer = shap.TreeExplainer(binary_model)
    # Sample for speed
    sample = X.sample(min(1000, len(X)), random_state=42)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-10:][::-1]
    metrics["top_shap_features"] = [available_features[i] for i in top_idx]
    print(f"\n  Top SHAP features: {metrics['top_shap_features'][:5]}")
except Exception as e:
    print(f"  SHAP computation skipped: {e}")

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 60)
print("  ✅ All models retrained and saved!")
print(f"  Features: {len(available_features)}")
print(f"  Training samples: {len(X)}")
print("=" * 60)
