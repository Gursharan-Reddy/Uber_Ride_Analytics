import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import os
import warnings

# --- FIX: SUPPRESS NOISY WARNINGS ---
# This ignores the "sklearn.utils.parallel.delayed" warnings from flooding the console
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

def train_demand_model(data_path='data/processed/model_data.parquet'):
    print("Loading training data...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None, 0, None

    df = pd.read_parquet(data_path)
    
    # 1. Define Features
    target = 'trip_count'
    # Exclude target, dates, and the intermediate string column 'temp_cat'
    exclude_cols = [target, 'date_only', 'tpep_pickup_datetime', 'temp_cat']
    
    # Select only numerical columns for training
    features = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Training on {len(features)} features: {features}")
    
    # Filter dataset to ensure only numeric data is passed
    X = df[features].select_dtypes(include=[np.number])
    y = df[target]

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train Random Forest
    print("Training Random Forest Regressor (this may take a moment)...")
    # n_jobs=-1 uses all processors. The warnings are suppressed now.
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluate
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    
    # Calculate RMSE manually to be safe across versions
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print("-" * 30)
    print(f"✅ MODEL TRAINED SUCCESSFULLY")
    print(f"R2 Score: {score:.4f}")
    print(f"RMSE:     {rmse:.2f}")
    print("-" * 30)

    # 5. Save Artifacts
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.pkl')
    # Save the feature names so Streamlit knows what inputs to create
    joblib.dump(X.columns.tolist(), 'outputs/feature_names.pkl')
    
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top 5 Drivers of Demand:")
    print(importance.head(5))

    return model, score, importance

if __name__ == "__main__":
    train_demand_model()