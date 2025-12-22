from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def train_demand_model(df):
    features = ['PULocationID', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_holiday', 'TAVG', 'TMAX', 'TMIN']
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df['trip_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    joblib.dump(model, 'outputs/model.pkl')
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, model.score(X_test, y_test), importance