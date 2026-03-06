#!/usr/bin/env python3
"""
Main Pipeline Orchestrator
Run the full data processing + ML training pipeline end-to-end.

Usage:
    python run_pipeline.py [--dataset-dir PATH] [--output-dir PATH] [--window-days N]
"""

import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.loader import load_all_plants
from src.data_processing.features import run_feature_engineering, get_feature_matrix
from src.ml.trainer import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Solar Inverter Failure Prediction Pipeline")
    parser.add_argument("--dataset-dir", default="dataset", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", default="models", help="Path to save trained models")
    parser.add_argument("--window-days", type=int, default=7, help="Prediction window in days")
    parser.add_argument("--save-processed", action="store_true", help="Save processed data to CSV")
    args = parser.parse_args()

    start_time = time.time()

    # ─── Step 1: Load Raw Data ────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE STEP 1: Loading Raw Data")
    print("="*60)
    df = load_all_plants(args.dataset_dir)

    # ─── Step 2: Feature Engineering ──────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE STEP 2: Feature Engineering")
    print("="*60)
    df = run_feature_engineering(df, target_window_days=args.window_days)

    if args.save_processed:
        processed_path = os.path.join("data", "processed", "featured_data.parquet")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_parquet(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")

    # ─── Step 3: Prepare Feature Matrix ───────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE STEP 3: Preparing Feature Matrix")
    print("="*60)
    X, y_binary, feature_names = get_feature_matrix(df, target_col="target_binary")
    _, y_multi, _ = get_feature_matrix(df, target_col="target_multiclass")

    # ─── Step 4: Train Models ─────────────────────────────────────────
    print("\n" + "="*60)
    print("PIPELINE STEP 4: Training Models")
    print("="*60)
    results = run_training_pipeline(X, y_binary, y_multi, feature_names, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"Models saved to: {args.output_dir}/")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
