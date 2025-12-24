import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_all_taxi_data, load_weather_data, load_zone_lookup
from src.feature_eng import preprocess_dates, merge_datasets, create_rich_features
from src.model_train import train_demand_model

def run_pipeline():
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # CHANGE: Added sample_frac=0.05 for 5% of data (Fast Mode)
    # To use 100% of data for final submission, change this to None
    print("Status: Starting Data Ingestion (Fast Mode - 5% Sample)")
    taxi = load_all_taxi_data('data/raw/', sample_frac=0.05)
    
    print("Status: Loading Weather and Zone metadata")
    weather = load_weather_data('data/raw/4191253.csv')
    zones = load_zone_lookup('data/raw/taxi_zone_lookup.csv')
    
    print("Status: Transforming Dates and Merging")
    taxi, weather = preprocess_dates(taxi, weather)
    combined = merge_datasets(taxi, weather)
    
    print("Status: Aggregating Origin-Destination Pairs (This may take a moment)")
    final_df = create_rich_features(combined)
    
    print("Status: Saving processed data to disk")
    final_df.to_parquet('data/processed/model_data.parquet', index=False)
    zones.to_csv('data/processed/zones.csv', index=False)
    
    print("Status: Training Random Forest Regressor")
    model, score, importance = train_demand_model(final_df)
    
    print(f"\nPipeline Complete!")
    print(f"Model R-squared Score: {score:.4f}")

if __name__ == "__main__":
    run_pipeline()