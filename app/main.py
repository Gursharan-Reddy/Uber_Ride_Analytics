import pandas as pd
import numpy as np
import joblib
import os
import sqlite3

# Define Paths relative to this file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "outputs", "feature_names.pkl")
DB_PATH = os.path.join(BASE_DIR, "data", "taxi_system.db")

class UberAnalyticsBackend:
    def __init__(self):
        self.model = self._load_model()
        self.features = self._load_features()
        
    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        return None

    def _load_features(self):
        if os.path.exists(FEATURES_PATH):
            return joblib.load(FEATURES_PATH)
        # Fallback defaults if file missing
        return ['PULocationID', 'hour', 'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'traffic_congestion', 'weather_main_Rain']

    def get_zone_data(self):
        """Connects to DB and returns Zone info"""
        if not os.path.exists(DB_PATH): return pd.DataFrame()
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        return pd.read_sql("SELECT * FROM zones", conn)

    def predict_demand(self, loc_id, hour, traffic_val, weather_val, date_obj):
        """
        Core Prediction Logic
        Calculates Cyclical Time Features -> Queries Model
        """
        if not self.model: return 0
        
        # 1. Feature Engineering (Must match training!)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_of_week = date_obj.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rain = 1 if weather_val == "Rain" else 0
        
        # 2. Build Input Array
        input_data = pd.DataFrame([{
            'PULocationID': loc_id,
            'hour': hour,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'traffic_congestion': traffic_val,
            'weather_main_Rain': is_rain
        }])
        
        # 3. Ensure Column Order
        input_data = input_data.reindex(columns=self.features, fill_value=0)
        
        # 4. Predict
        return max(0, self.model.predict(input_data)[0])

# Initialize backend instance
backend = UberAnalyticsBackend()