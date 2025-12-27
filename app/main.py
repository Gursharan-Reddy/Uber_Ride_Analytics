import sys
import os
import pandas as pd
# Add the project root to the python path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_all_taxi_data, load_weather_data, load_zone_lookup, load_spatial_data
from src.feature_eng import preprocess_dates, merge_datasets, create_rich_features
from src.model_train import train_demand_model

def run_pipeline():
    # Create directories if they don't exist
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    print("Step 1: Loading Raw Data...")
    # Update this path if your data is elsewhere
    taxi = load_all_taxi_data('data/raw/')
    weather = load_weather_data('data/raw/4191253.csv')
    zones = load_zone_lookup('data/raw/taxi_zone_lookup.csv')
    
    print("Step 2: Processing Spatial Data...")
    gdf = load_spatial_data()
    if gdf is not None:
        gdf.to_file('data/processed/zones.geojson', driver='GeoJSON')
    
    print("Step 3: Feature Engineering...")
    taxi, weather = preprocess_dates(taxi, weather)
    combined = merge_datasets(taxi, weather)
    final_df = create_rich_features(combined)
    
    # Save processed data to disk
    print("Status: Saving processed data to disk...")
    final_df.to_parquet('data/processed/model_data.parquet', index=False)
    zones.to_csv('data/processed/zones.csv', index=False)
    
    print("Step 4: Training ML Model...")
    # --- FIX IS HERE: Call function without arguments ---
    # It will automatically read 'data/processed/model_data.parquet'
    model, score, importance = train_demand_model() 
    
    print(f"Model Training Complete. R2 Score: {score:.4f}")

if __name__ == "__main__":
    run_pipeline()