import pandas as pd
import glob
import os
import geopandas as gpd

def load_all_taxi_data(raw_dir_path):
    """
    Loads all yellow taxi parquet files from the specified directory.
    """
    # Find all .parquet files in the folder (e.g., yellow_tripdata_2024-01.parquet)
    files = glob.glob(os.path.join(raw_dir_path, "yellow_tripdata_*.parquet"))
    
    if not files:
        print("Warning: No parquet files found in", raw_dir_path)
        return pd.DataFrame() # Return empty if no files found

    print(f"Found {len(files)} files to load...")
    df_list = []
    for f in files:
        print(f"Loading: {os.path.basename(f)}...")
        # Read file
        df = pd.read_parquet(f)
        # Optional: Sampling to speed up training for testing (remove .sample() for full training)
        df_list.append(df.sample(frac=0.05, random_state=42)) 
        
    return pd.concat(df_list, ignore_index=True)

def load_weather_data(file_path):
    """Loads weather CSV data."""
    return pd.read_csv(file_path)

def load_zone_lookup(file_path):
    """Loads the Taxi Zone Lookup CSV."""
    return pd.read_csv(file_path)

def load_spatial_data():
    """
    Fetches NYC Taxi Zone GeoJSON data for mapping.
    Uses the official NYC Open Data URL.
    """
    url = "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON"
    try:
        print("Fetching spatial data from NYC Open Data...")
        gdf = gpd.read_file(url)
        return gdf
    except Exception as e:
        print(f"Warning: Could not load spatial data. Error: {e}")
        return None