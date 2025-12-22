import pandas as pd
import glob
import os
import geopandas as gpd

def load_all_taxi_data(raw_dir_path):
    files = glob.glob(os.path.join(raw_dir_path, "yellow_tripdata_2024-*.parquet"))
    df_list = [pd.read_parquet(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def load_weather_data(file_path):
    return pd.read_csv(file_path)

def load_zone_lookup(file_path):
    return pd.read_csv(file_path)

def load_spatial_data():
    url = "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON"
    try:
        gdf = gpd.read_file(url)
        gdf['longitude'] = gdf.geometry.centroid.x
        gdf['latitude'] = gdf.geometry.centroid.y
        return gdf
    except:
        return None