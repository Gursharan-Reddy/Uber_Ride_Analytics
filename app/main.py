import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_all_taxi_data, load_weather_data, load_zone_lookup, load_spatial_data
from src.feature_eng import preprocess_dates, merge_datasets, create_rich_features
from src.model_train import train_demand_model

def run_pipeline():
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    taxi = load_all_taxi_data('data/raw/')
    weather = load_weather_data('data/raw/4191253.csv')
    zones = load_zone_lookup('data/raw/taxi_zone_lookup.csv')
    gdf = load_spatial_data()
    if gdf is not None:
        gdf.to_file('data/processed/zones.geojson', driver='GeoJSON')
    taxi, weather = preprocess_dates(taxi, weather)
    combined = merge_datasets(taxi, weather)
    final_df = create_rich_features(combined)
    final_df.to_parquet('data/processed/model_data.parquet', index=False)
    zones.to_csv('data/processed/zones.csv', index=False)
    model, score, importance = train_demand_model(final_df)
    print(f"Pipeline Complete. R2: {score:.4f}")

if __name__ == "__main__":
    run_pipeline()