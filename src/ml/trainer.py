"""
ML Training Module
Trains XGBoost models for inverter failure prediction with:
- Walk-forward (time-series aware) cross-validation
- Binary classification + multi-class
- SHAP explainability
- Isolation Forest anomaly detection as complementary signal
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional
import shap


# ─── XGBoost Training ─────────────────────────────────────────────────────────

def train_xgboost_binary(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    params: dict = None
) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classifier for failure prediction.

    Uses scale_pos_weight to handle class imbalance (failures are rare events).
    """
    if params is None:
        # Calculate class imbalance ratio
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_weight = neg_count / max(pos_count, 1)

        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_weight,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 50,
        }

    model = xgb.XGBClassifier(**params)

    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50
    )

    return model


def train_xgboost_multiclass(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
) -> xgb.XGBClassifier:
    """Train XGBoost multi-class model: 0=no risk, 1=degradation, 2=shutdown."""
    params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBClassifier(**params)

    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50
    )

    return model


# ─── Isolation Forest Anomaly Detection ───────────────────────────────────────

def train_anomaly_detector(X_train: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
    """Train Isolation Forest for anomaly detection as complementary signal."""
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def add_anomaly_scores(X: pd.DataFrame, anomaly_model: IsolationForest) -> pd.DataFrame:
    """Add anomaly scores from Isolation Forest to feature matrix."""
    X = X.copy()
    X["anomaly_score"] = anomaly_model.decision_function(X.drop(columns=["anomaly_score"], errors="ignore"))
    X["is_anomaly"] = (anomaly_model.predict(X.drop(columns=["anomaly_score", "is_anomaly"], errors="ignore")) == -1).astype(int)
    return X


# ─── Walk-Forward Cross-Validation ────────────────────────────────────────────

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_type: str = "binary"
) -> Dict:
    """
    Perform walk-forward (time-series aware) cross-validation.

    Returns dict with per-fold and aggregate metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"  Train: {len(X_train)} samples, Val: {len(X_val)} samples")
        print(f"  Train target dist: {y_train.value_counts().to_dict()}")

        if model_type == "binary":
            model = train_xgboost_binary(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            metrics = {
                "fold": fold_idx + 1,
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "auc": roc_auc_score(y_val, y_proba) if len(y_val.unique()) > 1 else 0,
            }
        else:
            model = train_xgboost_multiclass(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_val)

            metrics = {
                "fold": fold_idx + 1,
                "precision_macro": precision_score(y_val, y_pred, average='macro', zero_division=0),
                "recall_macro": recall_score(y_val, y_pred, average='macro', zero_division=0),
                "f1_macro": f1_score(y_val, y_pred, average='macro', zero_division=0),
            }

        print(f"  Metrics: {metrics}")
        fold_metrics.append(metrics)

    # Aggregate metrics
    result = {
        "per_fold": fold_metrics,
        "aggregate": {}
    }

    metric_keys = [k for k in fold_metrics[0].keys() if k != "fold"]
    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        result["aggregate"][f"{key}_mean"] = np.mean(values)
        result["aggregate"][f"{key}_std"] = np.std(values)

    print(f"\n{'='*60}")
    print(f"Aggregate Metrics:")
    for k, v in result["aggregate"].items():
        print(f"  {k}: {v:.4f}")

    return result


# ─── SHAP Explainability ──────────────────────────────────────────────────────

def compute_shap_values(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    max_samples: int = 5000
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values for model explainability.

    Returns:
        shap_values: SHAP values array
        top_features: List of top 5 most important feature names
    """
    # Subsample if needed for performance
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Get top features by mean absolute SHAP value
    if isinstance(shap_values, list):
        # Multi-class: use class 1 (positive class) values
        sv = np.abs(shap_values[1]).mean(axis=0) if len(shap_values) > 1 else np.abs(shap_values[0]).mean(axis=0)
    else:
        sv = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.Series(sv, index=X_sample.columns)
    top_features = feature_importance.nlargest(10).index.tolist()

    print(f"\nTop 10 SHAP Features:")
    for i, feat in enumerate(top_features):
        print(f"  {i+1}. {feat}: {feature_importance[feat]:.4f}")

    return shap_values, top_features


def get_shap_explanation_for_sample(
    model: xgb.XGBClassifier,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 5
) -> Dict:
    """Get SHAP explanation for a single prediction (for GenAI narrative input)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    if len(sv.shape) == 1:
        sv = sv.reshape(1, -1)

    # Get top contributing features for this sample
    abs_vals = np.abs(sv[0])
    top_idx = np.argsort(abs_vals)[-top_n:][::-1]

    explanation = {
        "top_features": [],
        "base_value": float(explainer.expected_value) if isinstance(explainer.expected_value, (int, float)) else float(explainer.expected_value[1]),
    }

    for idx in top_idx:
        explanation["top_features"].append({
            "feature": feature_names[idx],
            "value": float(X_sample.iloc[0, idx]),
            "shap_value": float(sv[0][idx]),
            "direction": "increases risk" if sv[0][idx] > 0 else "decreases risk"
        })

    return explanation


# ─── Full Training Pipeline ───────────────────────────────────────────────────

def run_training_pipeline(
    X: pd.DataFrame,
    y_binary: pd.Series,
    y_multi: pd.Series,
    feature_names: List[str],
    output_dir: str = "models"
) -> Dict:
    """
    Run the complete ML training pipeline:
    1. Train Isolation Forest for anomaly detection
    2. Add anomaly scores as features
    3. Walk-forward CV for binary classification
    4. Train final binary model on all data
    5. Walk-forward CV for multi-class
    6. Train final multi-class model
    7. Compute SHAP values
    8. Save models and metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # ─── Step 1: Anomaly Detection ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1: Training Anomaly Detector (Isolation Forest)")
    print("="*60)
    anomaly_model = train_anomaly_detector(X)
    X = add_anomaly_scores(X, anomaly_model)
    feature_names = list(X.columns)
    joblib.dump(anomaly_model, os.path.join(output_dir, "anomaly_model.joblib"))
    print("  Anomaly model saved.")

    # ─── Step 2: Binary Classification CV ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 2: Walk-Forward CV (Binary Classification)")
    print("="*60)
    cv_binary = walk_forward_cv(X, y_binary, n_splits=5, model_type="binary")
    results["cv_binary"] = cv_binary

    # ─── Step 3: Train Final Binary Model ─────────────────────────────
    print("\n" + "="*60)
    print("STEP 3: Training Final Binary Model")
    print("="*60)
    # Use last 20% as hold-out
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_binary.iloc[:split_idx], y_binary.iloc[split_idx:]

    binary_model = train_xgboost_binary(X_train, y_train, X_test, y_test)

    # Evaluate on hold-out
    y_pred = binary_model.predict(X_test)
    y_proba = binary_model.predict_proba(X_test)[:, 1]

    holdout_metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0),
    }
    results["holdout_binary"] = holdout_metrics
    print(f"\nHold-out Metrics: {holdout_metrics}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    joblib.dump(binary_model, os.path.join(output_dir, "binary_model.joblib"))

    # ─── Step 4: Multi-Class CV ───────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4: Walk-Forward CV (Multi-Class)")
    print("="*60)
    cv_multi = walk_forward_cv(X, y_multi, n_splits=5, model_type="multiclass")
    results["cv_multiclass"] = cv_multi

    # ─── Step 5: Train Final Multi-Class Model ────────────────────────
    print("\n" + "="*60)
    print("STEP 5: Training Final Multi-Class Model")
    print("="*60)
    y_train_m, y_test_m = y_multi.iloc[:split_idx], y_multi.iloc[split_idx:]
    multi_model = train_xgboost_multiclass(X_train, y_train_m, X_test, y_test_m)
    joblib.dump(multi_model, os.path.join(output_dir, "multiclass_model.joblib"))

    # ─── Step 6: SHAP Values ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6: Computing SHAP Values")
    print("="*60)
    shap_values, top_features = compute_shap_values(binary_model, X_test)
    results["top_shap_features"] = top_features

    # ─── Step 7: Save Results ─────────────────────────────────────────
    # Save feature names
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.joblib"))

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All models and metrics saved to {output_dir}/")
    print(f"{'='*60}")

    return results
