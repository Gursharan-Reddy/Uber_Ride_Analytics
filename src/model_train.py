from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def train_demand_model(df):
    # Added DOLocationID to the feature set
    base_features = [
        'PULocationID', 'DOLocationID', 'hour', 'day_of_week', 
        'month', 'is_weekend', 'is_holiday', 'is_rainy', 'is_freezing'
    ]
    weather_cols = [c for c in df.columns if c.startswith('weather_')]
    features = base_features + weather_cols
    
    X = df[features]
    y = df['trip_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'outputs/model.pkl')
    joblib.dump(features, 'outputs/feature_names.pkl')
    
    return model, model.score(X_test, y_test), pd.Series(model.feature_importances_, index=features)