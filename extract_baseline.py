import os
import json
import pandas as pd
from src.data_processing.loader import load_all_plants
from src.data_processing.features import run_feature_engineering, get_feature_matrix

def main():
    print("Loading data...")
    df = load_all_plants("data/raw_unzipped")
    print("Running feature engineering...")
    df = run_feature_engineering(df, target_window_days=7)
    print("Getting feature matrix...")
    X, y_binary, feature_names = get_feature_matrix(df, target_col="target_binary")
    
    # Filter for healthy inverters (target_binary == 0)
    print(f"Total rows: {len(X)}")
    healthy_mask = (y_binary == 0)
    X_healthy = X[healthy_mask]
    print(f"Healthy rows: {len(X_healthy)}")
    
    # Calculate medians for healthy rows
    medians = X_healthy.median().to_dict()
    
    with open("healthy_baseline.json", "w") as f:
        json.dump(medians, f, indent=2)
    print("Saved to healthy_baseline.json")

if __name__ == "__main__":
    main()
