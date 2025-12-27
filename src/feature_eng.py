import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

def preprocess_dates(taxi_df, weather_df):
    # 1. Process Taxi Dates
    taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
    taxi_df['date_only'] = taxi_df['tpep_pickup_datetime'].dt.normalize()
    taxi_df['hour'] = taxi_df['tpep_pickup_datetime'].dt.hour
    
    # 2. Process Weather Dates
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    
    # 3. SAFETY CHECK: Ensure critical columns exist
    # If PRCP (Rain) is missing, assume 0 (No Rain)
    if 'PRCP' not in weather_df.columns:
        weather_df['PRCP'] = 0.0
    else:
        weather_df['PRCP'] = weather_df['PRCP'].fillna(0)
        
    # If TAVG (Avg Temp) is missing, try to calculate from Max/Min, or default to 60°F
    if 'TAVG' not in weather_df.columns:
        if 'TMAX' in weather_df.columns and 'TMIN' in weather_df.columns:
            weather_df['TAVG'] = (weather_df['TMAX'] + weather_df['TMIN']) / 2
        else:
            weather_df['TAVG'] = 60.0 # Default fallback
    else:
        weather_df['TAVG'] = weather_df['TAVG'].fillna(60.0)
            
    return taxi_df, weather_df

def merge_datasets(taxi_df, weather_df):
    # Left join to keep all taxi trips, adding weather where dates match
    combined = pd.merge(taxi_df, weather_df, left_on='date_only', right_on='DATE', how='left')
    return combined

def create_rich_features(df):
    # 1. Aggregate Taxi Demand (Count trips per Zone per Hour)
    zone_hourly_demand = df.groupby(['date_only', 'hour', 'PULocationID']).size().reset_index(name='trip_count')
    
    # 2. Extract Weather (It's duplicated per row, so just get unique values per date)
    weather_cols = ['date_only']
    if 'TAVG' in df.columns: weather_cols.append('TAVG')
    if 'PRCP' in df.columns: weather_cols.append('PRCP')
    
    # Drop duplicates to get one row per date
    daily_weather = df[weather_cols].drop_duplicates(subset=['date_only'])
    
    # Merge Demand with Weather
    final_df = pd.merge(zone_hourly_demand, daily_weather, on='date_only', how='left')
    
    # Safety Fill for any missing values after merge
    if 'TAVG' not in final_df.columns: final_df['TAVG'] = 60.0
    if 'PRCP' not in final_df.columns: final_df['PRCP'] = 0.0
    final_df['TAVG'] = final_df['TAVG'].fillna(60.0)
    final_df['PRCP'] = final_df['PRCP'].fillna(0.0)

    # 3. Date Features
    final_df['day_of_week'] = final_df['date_only'].dt.dayofweek
    final_df['month'] = final_df['date_only'].dt.month
    final_df['is_weekend'] = final_df['day_of_week'].isin([5, 6]).astype(int)
    
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=final_df['date_only'].min(), end=final_df['date_only'].max())
    final_df['is_holiday'] = final_df['date_only'].isin(holidays).astype(int)

    # 4. WEATHER CATEGORIZATION
    # This aligns the Training Data with your Dashboard Logic
    conditions = [
        (final_df['TAVG'] <= 32),
        (final_df['TAVG'] > 32) & (final_df['TAVG'] <= 50),
        (final_df['TAVG'] > 50) & (final_df['TAVG'] <= 72),
        (final_df['TAVG'] > 72) & (final_df['TAVG'] <= 85),
        (final_df['TAVG'] > 85)
    ]
    choices = ['Freezing', 'Cold', 'Mild', 'Warm', 'Hot']
    final_df['temp_cat'] = np.select(conditions, choices, default='Mild')

    # One-Hot Encode categories
    dummies = pd.get_dummies(final_df['temp_cat'], prefix='weather')
    final_df = pd.concat([final_df, dummies], axis=1)

    # Add Rain/Clear columns based on PRCP
    final_df['weather_Rain'] = (final_df['PRCP'] > 0).astype(int)
    final_df['weather_Clear'] = (final_df['PRCP'] == 0).astype(int)
    final_df['weather_Snow'] = 0 

    # Ensure all expected columns exist (fill with 0 if missing)
    expected_cols = ['weather_Freezing', 'weather_Cold', 'weather_Mild', 'weather_Warm', 'weather_Hot']
    for col in expected_cols:
        if col not in final_df.columns:
            final_df[col] = 0

    return final_df