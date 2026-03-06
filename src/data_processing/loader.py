"""
Data Loading Module
Handles loading raw CSV telemetry data from heterogeneous plant schemas
and normalizing into a unified per-inverter DataFrame.

Optimized version: aggregates to hourly data during loading to keep
memory usage manageable (~500K rows instead of 6M).
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List


def _detect_inverter_count(columns: List[str]) -> int:
    """Detect how many inverters are in the dataset based on column names."""
    max_idx = -1
    for col in columns:
        if col.startswith("inverters["):
            try:
                idx = int(col.split("[")[1].split("]")[0])
                max_idx = max(max_idx, idx)
            except (ValueError, IndexError):
                continue
    return max_idx + 1 if max_idx >= 0 else 0


def load_single_csv(filepath: str, plant_id: str) -> pd.DataFrame:
    """
    Load a single CSV file and normalize it into per-inverter rows.
    Each inverter in the wide-format CSV becomes its own set of rows.

    Returns a DataFrame with unified columns per inverter.
    """
    print(f"  Loading {os.path.basename(filepath)}...")

    df = pd.read_csv(filepath, low_memory=False)
    columns = list(df.columns)
    n_inverters = _detect_inverter_count(columns)
    print(f"    Detected {n_inverters} inverters, {len(df)} rows")

    # Parse datetime
    if "timestampDate" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestampDate"], errors='coerce')
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit='ms', errors='coerce')
    df = df.dropna(subset=["datetime"])

    # MAC for identification
    mac = df["mac"].iloc[0] if "mac" in df.columns else os.path.basename(filepath).replace(".csv", "")

    # ─── Extract common meter/sensor columns ──────────────────────────
    common_map = {}
    for c in columns:
        if c.startswith("sensors[0]."):
            common_map[c] = c.replace("sensors[0].", "")
        elif c.startswith("meters[0]."):
            common_map[c] = c.replace("meters[0].", "meter_")

    common_cols_available = [c for c in common_map.keys() if c in columns]
    common_data = df[["datetime"] + common_cols_available].rename(columns=common_map)

    # Convert common cols to numeric
    for col in common_data.columns:
        if col != "datetime":
            common_data[col] = pd.to_numeric(common_data[col], errors='coerce')

    # ─── Extract per-inverter data ────────────────────────────────────
    all_inverter_dfs = []

    for inv_idx in range(n_inverters):
        prefix = f"inverters[{inv_idx}]."
        inv_cols = {c: c.replace(prefix, "inv_") for c in columns if c.startswith(prefix)}

        if not inv_cols:
            continue

        inv_data = df[list(inv_cols.keys())].rename(columns=inv_cols)
        for col in inv_data.columns:
            inv_data[col] = pd.to_numeric(inv_data[col], errors='coerce')

        # ─── Extract SMU data for this inverter ───────────────────────
        smu_prefix = f"smu[{inv_idx}]." if f"smu[{inv_idx}].string1" in columns else "smu[0]."
        smu_cols = [c for c in columns if c.startswith(smu_prefix) and not c.endswith(".id")]

        smu_features = pd.DataFrame(index=df.index)
        if smu_cols:
            smu_data = df[smu_cols].apply(pd.to_numeric, errors='coerce')
            smu_features["smu_string_mean"] = smu_data.mean(axis=1)
            smu_features["smu_string_std"] = smu_data.std(axis=1)
            smu_features["smu_num_zero"] = (smu_data == 0).sum(axis=1)
            smu_features["smu_total_strings"] = len(smu_cols)

        # Combine everything
        combined = pd.concat([common_data.reset_index(drop=True),
                              inv_data.reset_index(drop=True),
                              smu_features.reset_index(drop=True)], axis=1)
        combined["plant_id"] = plant_id
        combined["logger_mac"] = mac
        combined["inverter_idx"] = inv_idx
        combined["inverter_id"] = f"{plant_id}_{mac}_INV{inv_idx}"

        all_inverter_dfs.append(combined)

    if not all_inverter_dfs:
        return pd.DataFrame()

    result = pd.concat(all_inverter_dfs, ignore_index=True)
    print(f"    -> {len(result)} per-inverter rows")
    return result


def _aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 5-min data to hourly for each inverter.
    Keeps mean for continuous signals, max for alarms, sum for energy.
    """
    df = df.copy()
    df["hour_bucket"] = df["datetime"].dt.floor("h")

    # Separate metadata columns from numeric
    meta_cols = ["plant_id", "logger_mac", "inverter_idx", "inverter_id"]
    group_cols = ["inverter_id", "hour_bucket"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in meta_cols + ["inverter_idx"]]

    # Build aggregation rules
    agg_dict = {}
    for col in numeric_cols:
        if "alarm" in col:
            agg_dict[col] = "max"  # any alarm in the hour
        elif "kwh_total" in col or "kwh_today" in col:
            agg_dict[col] = "last"  # cumulative: take latest
        elif "op_state" in col:
            agg_dict[col] = "max"  # worst state in the hour
        else:
            agg_dict[col] = "mean"  # average for continuous signals

    hourly = df.groupby(group_cols).agg(agg_dict).reset_index()

    # Restore metadata
    meta = df.groupby("inverter_id")[meta_cols].first().reset_index(drop=True)
    meta_lookup = df.groupby("inverter_id")[meta_cols].first()

    hourly = hourly.rename(columns={"hour_bucket": "datetime"})

    # Merge metadata back
    for col in meta_cols:
        if col != "inverter_id":
            hourly[col] = hourly["inverter_id"].map(meta_lookup[col])

    return hourly


def load_all_plants(dataset_dir: str, hourly: bool = True) -> pd.DataFrame:
    """
    Load all CSV files from all plant directories.
    Optionally aggregates to hourly intervals to save memory.
    """
    all_dfs = []

    plant_dirs = sorted(glob.glob(os.path.join(dataset_dir, "Plant *")))
    for plant_dir in plant_dirs:
        plant_name = os.path.basename(plant_dir).replace(" ", "_")
        print(f"\nProcessing {plant_name}...")

        csv_files = sorted(glob.glob(os.path.join(plant_dir, "*.csv")))
        for csv_file in csv_files:
            df = load_single_csv(csv_file, plant_name)
            if not df.empty:
                if hourly:
                    df = _aggregate_hourly(df)
                all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No data loaded!")

    unified = pd.concat(all_dfs, ignore_index=True)
    unified = unified.sort_values(["inverter_id", "datetime"]).reset_index(drop=True)
    if hourly:
        print(f"Hourly total: {len(unified)} rows")
    else:
        print(f"\nRaw total: {len(unified)} rows, {unified['inverter_id'].nunique()} inverters")

    print(f"Date range: {unified['datetime'].min()} to {unified['datetime'].max()}")
    return unified


if __name__ == "__main__":
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")
    df = load_all_plants(dataset_dir, hourly=True)
    print(f"\nColumns: {list(df.columns)}")
    print(df.head())
