import pandas as pd
import glob
import os

def load_all_taxi_data(raw_dir_path, sample_frac=None):
    files = glob.glob(os.path.join(raw_dir_path, "yellow_tripdata_2024-*.parquet"))
    df_list = []
    
    for f in files:
        print(f"Loading: {os.path.basename(f)}...")
        temp_df = pd.read_parquet(f)
        
        # If sampling is enabled, take only a portion of the data
        if sample_frac:
            temp_df = temp_df.sample(frac=sample_frac, random_state=42)
            
        df_list.append(temp_df)
        
    print("Combining all files into memory...")
    return pd.concat(df_list, ignore_index=True)

def load_weather_data(file_path):
    return pd.read_csv(file_path)

def load_zone_lookup(file_path):
    return pd.read_csv(file_path)