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

def categorize_temp(temp):
    if temp <= 32: return 'Freezing'
    if temp <= 50: return 'Cold'
    if temp <= 72: return 'Mild'
    if temp <= 85: return 'Warm'
    return 'Hot'

def create_rich_features(df):
    available_cols = ['TAVG', 'TMAX', 'TMIN', 'PRCP']
    agg_dict = {col: 'first' for col in available_cols if col in df.columns}
    
    daily_weather = df.groupby('date_only').agg(agg_dict).reset_index()
    
    # Updated: Now grouping by BOTH Pickup and Destination IDs
    route_hourly = df.groupby(['date_only', 'hour', 'PULocationID', 'DOLocationID']).size().reset_index(name='trip_count')
    
    final_df = pd.merge(route_hourly, daily_weather, on='date_only', how='left')
    
    for col in agg_dict.keys():
        final_df[col] = final_df[col].fillna(final_df[col].mean())

    final_df['temp_category'] = final_df['TAVG'].apply(categorize_temp)
    final_df['is_rainy'] = (final_df['PRCP'] > 0.1).astype(int) if 'PRCP' in final_df.columns else 0
    final_df['is_freezing'] = (final_df['TMIN'] <= 32).astype(int)
    
    final_df['day_of_week'] = final_df['date_only'].dt.dayofweek
    final_df['month'] = final_df['date_only'].dt.month
    final_df['is_weekend'] = final_df['day_of_week'].isin([5, 6]).astype(int)
    
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=final_df['date_only'].min(), end=final_df['date_only'].max())
    final_df['is_holiday'] = final_df['date_only'].isin(holidays).astype(int)
    
    final_df = pd.get_dummies(final_df, columns=['temp_category'], prefix='weather')
    
    return final_df