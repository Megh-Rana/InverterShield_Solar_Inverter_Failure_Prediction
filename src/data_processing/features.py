"""
Feature Engineering Module
Transforms the unified per-inverter DataFrame into ML-ready features.

Optimized for HOURLY data (after aggregation from 5-min intervals).
Window sizes are in hours, not 5-minute intervals.
"""

import pandas as pd
import numpy as np


# ─── Rolling window sizes (in hours, since data is hourly) ───────────────────
WINDOWS = {
    "6h": 6,
    "24h": 24,
    "7d": 168,    # 7 * 24
}

# Columns to compute rolling statistics for
ROLLING_COLS = [
    "inv_power", "inv_temp",
    "ambient_temp", "meter_freq", "meter_pf",
    "meter_v_r", "meter_v_y", "meter_v_b",
    "smu_string_mean",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from datetime column."""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    return df


def add_rolling_features(df: pd.DataFrame, group_col: str = "inverter_id") -> pd.DataFrame:
    """Add rolling window statistics per inverter."""
    df = df.copy()

    for col in ROLLING_COLS:
        if col not in df.columns:
            continue

        numeric_col = pd.to_numeric(df[col], errors='coerce')

        for window_name, window_size in WINDOWS.items():
            grouped_rolling = numeric_col.groupby(df[group_col]).rolling(
                window=window_size, min_periods=max(1, window_size // 4)
            )

            suffix = f"{col}_{window_name}"
            df[f"{suffix}_mean"] = grouped_rolling.mean().reset_index(level=0, drop=True)
            df[f"{suffix}_std"] = grouped_rolling.std().reset_index(level=0, drop=True)

    return df


def add_alarm_features(df: pd.DataFrame, group_col: str = "inverter_id") -> pd.DataFrame:
    """Add alarm-based features."""
    df = df.copy()

    if "inv_alarm_code" not in df.columns:
        df["alarm_active"] = 0
        df["alarm_count_24h"] = 0
        df["alarm_count_7d"] = 0
        df["hours_since_last_alarm"] = 9999
        return df

    alarm_col = pd.to_numeric(df["inv_alarm_code"], errors='coerce').fillna(0)
    df["alarm_active"] = (alarm_col > 0).astype(int)

    df["alarm_count_24h"] = df["alarm_active"].groupby(df[group_col]).rolling(
        window=24, min_periods=1
    ).sum().reset_index(level=0, drop=True)

    df["alarm_count_7d"] = df["alarm_active"].groupby(df[group_col]).rolling(
        window=168, min_periods=1
    ).sum().reset_index(level=0, drop=True)

    # Time since last alarm (vectorized)
    df["hours_since_last_alarm"] = 9999.0
    for inv_id, group in df.groupby(group_col):
        mask = group["alarm_active"] == 1
        if mask.any():
            alarm_timestamps = group["datetime"].where(mask).ffill()
            hours_since = (group["datetime"] - alarm_timestamps).dt.total_seconds() / 3600
            df.loc[group.index, "hours_since_last_alarm"] = hours_since.fillna(9999.0)

    return df


def add_kpi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed KPI features."""
    df = df.copy()

    inv_power = pd.to_numeric(df.get("inv_power", pd.Series(dtype=float)), errors='coerce').fillna(0)

    # Voltage imbalance across phases
    v_cols = ["meter_v_r", "meter_v_y", "meter_v_b"]
    available_v = [c for c in v_cols if c in df.columns]
    if len(available_v) == 3:
        v_values = df[available_v].apply(pd.to_numeric, errors='coerce')
        v_mean = v_values.mean(axis=1)
        v_max_dev = v_values.sub(v_mean, axis=0).abs().max(axis=1)
        df["voltage_imbalance"] = np.where(v_mean > 0, v_max_dev / v_mean, 0)
    else:
        df["voltage_imbalance"] = 0

    # Power factor deviation
    if "meter_pf" in df.columns:
        pf = pd.to_numeric(df["meter_pf"], errors='coerce').fillna(1.0)
        df["pf_deviation"] = (1.0 - pf.abs()).abs()
    else:
        df["pf_deviation"] = 0

    # Frequency deviation from 50 Hz
    if "meter_freq" in df.columns:
        freq = pd.to_numeric(df["meter_freq"], errors='coerce').fillna(50.0)
        df["freq_deviation"] = (freq - 50.0).abs()
    else:
        df["freq_deviation"] = 0

    # Power ratio vs 24h average
    if "inv_power_24h_mean" in df.columns:
        avg_power = pd.to_numeric(df["inv_power_24h_mean"], errors='coerce')
        df["power_ratio_vs_24h"] = np.where(avg_power > 10, inv_power / avg_power, 1.0)
    else:
        df["power_ratio_vs_24h"] = 1.0

    # SMU zero-string fraction
    if "smu_num_zero" in df.columns and "smu_total_strings" in df.columns:
        total = pd.to_numeric(df["smu_total_strings"], errors='coerce').fillna(1)
        zeros = pd.to_numeric(df["smu_num_zero"], errors='coerce').fillna(0)
        df["smu_zero_fraction"] = zeros / total.replace(0, 1)
    else:
        df["smu_zero_fraction"] = 0

    return df


def add_target_variable(df: pd.DataFrame, window_days: int = 7,
                        group_col: str = "inverter_id") -> pd.DataFrame:
    """
    Create target: will inverter fail in next `window_days` days?

    Multi-class: 0=No Risk, 1=Degradation, 2=Shutdown
    """
    df = df.copy()
    window_hours = window_days * 24  # hourly data

    inv_power = pd.to_numeric(df.get("inv_power", pd.Series(dtype=float)), errors='coerce').fillna(0)
    alarm_code = pd.to_numeric(df.get("inv_alarm_code", pd.Series(dtype=float)), errors='coerce').fillna(0)

    is_daytime = df.get("is_daytime", pd.Series(0, index=df.index))

    # Shutdown: alarm OR zero power during daytime
    shutdown_event = ((alarm_code > 0) | ((inv_power == 0) & (is_daytime == 1))).astype(int)

    # Degradation: power < 50% of rolling avg during expected production
    if "inv_power_24h_mean" in df.columns:
        avg_power = pd.to_numeric(df["inv_power_24h_mean"], errors='coerce').fillna(0)
        degradation_event = (
            (inv_power > 0) & (avg_power > 100) & (inv_power < avg_power * 0.5)
        ).astype(int)
    else:
        degradation_event = pd.Series(0, index=df.index)

    # Forward-looking target using reverse rolling sum
    df["target_binary"] = 0
    df["target_multiclass"] = 0

    for inv_id, group in df.groupby(group_col):
        idx = group.index

        shutdown_future = shutdown_event.loc[idx].iloc[::-1].rolling(
            window=window_hours, min_periods=1
        ).sum().iloc[::-1]

        degradation_future = degradation_event.loc[idx].iloc[::-1].rolling(
            window=window_hours, min_periods=1
        ).sum().iloc[::-1]

        has_future_event = (shutdown_future > 0) | (degradation_future > 0)
        df.loc[idx, "target_binary"] = has_future_event.astype(int).values

        multiclass = pd.Series(0, index=idx)
        multiclass[degradation_future > 0] = 1
        multiclass[shutdown_future > 0] = 2
        df.loc[idx, "target_multiclass"] = multiclass.values

    return df


def run_feature_engineering(df: pd.DataFrame, target_window_days: int = 7) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print("Adding time features...")
    df = add_time_features(df)

    print("Adding rolling features...")
    df = add_rolling_features(df)

    print("Adding alarm features...")
    df = add_alarm_features(df)

    print("Adding KPI features...")
    df = add_kpi_features(df)

    print("Adding target variable...")
    df = add_target_variable(df, window_days=target_window_days)

    initial_len = len(df)
    df = df.dropna(subset=["target_binary"])
    print(f"Dropped {initial_len - len(df)} rows with NaN targets")

    print(f"\nFeature engineering complete!")
    print(f"  Total rows: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Target (binary): {df['target_binary'].value_counts().to_dict()}")
    print(f"  Target (multi):  {df['target_multiclass'].value_counts().to_dict()}")

    return df


# ─── Columns to exclude from features ────────────────────────────────────────
EXCLUDE_COLS = [
    "datetime", "timestampDate", "plant_id", "logger_mac",
    "inverter_idx", "inverter_id",
    "target_binary", "target_multiclass",
    "inv_alarm_code", "inv_op_state",  # leak target
    "alarm_active",  # leak target
]


def get_feature_matrix(df: pd.DataFrame, target_col: str = "target_binary"):
    """Extract feature matrix X and target vector y."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[target_col].copy()

    print(f"Feature matrix: {X.shape}, Target dist: {y.value_counts().to_dict()}")
    return X, y, feature_cols
