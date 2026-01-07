import sqlite3
import pandas as pd
import os

def init_database():
    # Define paths relative to the project root
    db_path = os.path.join('data', 'taxi_system.db')
    parquet_path = os.path.join('data', 'processed', 'model_data.parquet')
    zones_path = os.path.join('data', 'processed', 'zones.csv')
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    print("Database connection established.")

    # Migrate processed model data to SQLite
    if os.path.exists(parquet_path):
        print(f"Reading: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df.to_sql('model_data', conn, if_exists='replace', index=False)
        print("Model data migrated to SQLite table: model_data")
        
        # Optimize performance with indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hour ON model_data (hour)")
    else:
        print(f"Error: Parquet file not found at {parquet_path}")

    # Migrate zones data to SQLite
    if os.path.exists(zones_path):
        print(f"Reading: {zones_path}")
        zones = pd.read_csv(zones_path)
        zones.to_sql('zones', conn, if_exists='replace', index=False)
        print("Zone data migrated to SQLite table: zones")
    else:
        print(f"Error: Zones file not found at {zones_path}")

    conn.close()
    print("Database initialization process finished.")

if __name__ == "__main__":
    init_database()