import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import glob

# 1. SETUP PATHS
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "outputs", "feature_names.pkl")

print("🚀 Starting Training Pipeline using REAL PARQUET DATA...")

# 2. LOAD PARQUET FILES
parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
if not parquet_files:
    print(f"❌ No .parquet files found in {DATA_DIR}. Please add NYC Taxi data.")
    # Fallback to synthetic if empty (just to prevent crash during testing)
    df = pd.DataFrame()
else:
    print(f"📂 Found {len(parquet_files)} parquet files. Loading...")
    df_list = []
    for f in parquet_files:
        try:
            # Read only necessary columns to save memory
            temp = pd.read_parquet(f, columns=['tpep_pickup_datetime', 'PULocationID'])
            df_list.append(temp)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            
    if df_list:
        df = pd.concat(df_list)
    else:
        df = pd.DataFrame()

# 3. PREPROCESS & FEATURE ENGINEERING
if not df.empty:
    print(f"📊 Processing {len(df)} rides...")
    
    # Extract Time Features
    df['pickup_dt'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['pickup_dt'].dt.hour
    df['day_of_week'] = df['pickup_dt'].dt.dayofweek
    df['month'] = df['pickup_dt'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # AGGREGATE: Count rides per Location per Hour (This creates the Demand Target)
    # This naturally captures the "M" shape (Rush Hours)
    agg_df = df.groupby(['PULocationID', 'hour', 'day_of_week', 'is_weekend']).size().reset_index(name='trip_count')
    
    # 4. ENHANCE WITH CYCLICAL TIME & WEATHER (Simulated for Model Robustness)
    # Even with real data, we add sin/cos so the model understands Time Cycles
    agg_df['hour_sin'] = np.sin(2 * np.pi * agg_df['hour'] / 24)
    agg_df['hour_cos'] = np.cos(2 * np.pi * agg_df['hour'] / 24)
    
    # Add dummy weather/traffic columns (since historical weather might not be in parquet)
    # We train with "Clear" and "Average Traffic" as baseline 0
    agg_df['traffic_congestion'] = 0.5 
    agg_df['weather_main_Rain'] = 0 
    
    # 5. TRAIN MODEL
    print("🧠 Training Random Forest on Real Aggregated Data...")
    
    features = ['PULocationID', 'hour', 'hour_sin', 'hour_cos', 'day_of_week', 
                'is_weekend', 'traffic_congestion', 'weather_main_Rain']
    
    X = agg_df[features]
    y = agg_df['trip_count']
    
    # Use robust parameters to prevent overfitting
    model = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=4, random_state=42)
    model.fit(X, y)
    
    # 6. SAVE
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURE_NAMES_PATH)
    print(f"✅ Success! Model trained on {len(agg_df)} aggregated time-slots and saved.")

else:
    print("❌ Critical: No data available to train.")