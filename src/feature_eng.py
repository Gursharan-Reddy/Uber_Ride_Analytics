import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

def preprocess_dates(taxi_df, weather_df):
    taxi_df['tpep_pickup_datetime'] = pd.to_datetime(taxi_df['tpep_pickup_datetime'])
    taxi_df['date_only'] = taxi_df['tpep_pickup_datetime'].dt.normalize()
    taxi_df['hour'] = taxi_df['tpep_pickup_datetime'].dt.hour
    weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])
    return taxi_df, weather_df

def merge_datasets(taxi_df, weather_df):
    return pd.merge(taxi_df, weather_df, left_on='date_only', right_on='DATE', how='left')

def create_rich_features(df):
    # Group by Zone and Hour
    daily_weather = df.groupby('date_only').agg({
        'TAVG': 'first', 'TMAX': 'first', 'TMIN': 'first'
    }).reset_index()
    
    zone_hourly = df.groupby(['date_only', 'hour', 'PULocationID']).size().reset_index(name='trip_count')
    
    final_df = pd.merge(zone_hourly, daily_weather, on='date_only', how='left')
    
    # Fill missing weather
    for col in ['TAVG', 'TMAX', 'TMIN']:
        final_df[col] = final_df[col].fillna(final_df[col].mean())

    final_df['day_of_week'] = final_df['date_only'].dt.dayofweek
    final_df['month'] = final_df['date_only'].dt.month
    final_df['is_weekend'] = final_df['day_of_week'].isin([5, 6]).astype(int)
    
    # Add Holiday flag
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=final_df['date_only'].min(), end=final_df['date_only'].max())
    final_df['is_holiday'] = final_df['date_only'].isin(holidays).astype(int)
    
    return final_df