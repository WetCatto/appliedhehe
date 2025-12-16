import pandas as pd
import os

# Define dtypes for memory efficiency during load
dtypes = {
    'YEAR': 'int16', 'MONTH': 'int8', 'DAY': 'int8', 'DAY_OF_WEEK': 'int8',
    'AIRLINE': 'category', 'FLIGHT_NUMBER': 'int16', 'TAIL_NUMBER': 'category',
    'ORIGIN_AIRPORT': 'category', 'DESTINATION_AIRPORT': 'category',
    'SCHEDULED_DEPARTURE': 'int16', 'DEPARTURE_DELAY': 'float32',
    'TAXI_OUT': 'float32', 'WHEELS_OFF': 'float32', 'SCHEDULED_TIME': 'float32',
    'ELAPSED_TIME': 'float32', 'AIR_TIME': 'float32', 'DISTANCE': 'int16',
    'WHEELS_ON': 'float32', 'TAXI_IN': 'float32', 'SCHEDULED_ARRIVAL': 'int16',
    'ARRIVAL_TIME': 'float32', 'ARRIVAL_DELAY': 'float32', 'DIVERTED': 'int8',
    'CANCELLED': 'int8', 'CANCELLATION_REASON': 'category', # optimized
    'AIR_SYSTEM_DELAY': 'float32', 'SECURITY_DELAY': 'float32',
    'AIRLINE_DELAY': 'float32', 'LATE_AIRCRAFT_DELAY': 'float32',
    'WEATHER_DELAY': 'float32'
}

print("Loading CSV...")
try:
    df = pd.read_csv('flights.csv', dtype=dtypes, low_memory=False)
    
    
    # Drop unused columns to save space
    drop_cols = ['FLIGHT_NUMBER', 'TAIL_NUMBER', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'SCHEDULED_TIME', 'ELAPSED_TIME']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"Dropped {len(drop_cols)} columns.") # unused
    
    # optimize types further to save space
    float_cols = [
        'DEPARTURE_DELAY', 'TAXI_OUT', 'AIR_TIME',
        'ARRIVAL_TIME', 'ARRIVAL_DELAY', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 
        'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'
    ]
    
    for c in float_cols:
        # Convert to nullable Int16 if ranges permit
        # most delays are small.
        # Check range roughly or just try safe cast
        # Using Int32 to be safe against overflows, still smaller than float64/32 potentially when compressed?
        # Actually standard integer compression in parquet is very good.
        df[c] = df[c].astype("Int32") # Nullable Integer
        
    print("Optimized types.")

    # Try gzip for better compression to hit < 100MB
    df.to_parquet('data/processed/flights.parquet', compression='gzip', index=False)
    
    csv_size = os.path.getsize('flights.csv') / (1024*1024)
    # Display file sizes
    pq_size = os.path.getsize('data/processed/flights.parquet') / (1024*1024)
    
    print("Conversion Successful.")
    print(f"Original: {csv_size:.2f} MB")
    print(f"Parquet:  {pq_size:.2f} MB")
    
except Exception as e:
    print(f"Error: {e}")
