import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """
    Loads and merges the flights, airlines, and airports data.
    Optimizes memory usage by specifying data types.
    """
    try:
        # Load flights data
        # Optimization: Read specific columns only if needed, but for now we read all relevant ones
        # and specify types to save memory.
        dtypes = {
            'YEAR': 'int16',
            'MONTH': 'int8',
            'DAY': 'int8',
            'DAY_OF_WEEK': 'int8',
            'AIRLINE': 'string',
            'FLIGHT_NUMBER': 'int16',
            'TAIL_NUMBER': 'string',
            'ORIGIN_AIRPORT': 'string',
            'DESTINATION_AIRPORT': 'string',
            'SCHEDULED_DEPARTURE': 'int16',
            'DEPARTURE_DELAY': 'float32',
            'TAXI_OUT': 'float32',
            'WHEELS_OFF': 'float32',
            'SCHEDULED_TIME': 'float32',
            'ELAPSED_TIME': 'float32',
            'AIR_TIME': 'float32',
            'DISTANCE': 'int16',
            'WHEELS_ON': 'float32',
            'TAXI_IN': 'float32',
            'SCHEDULED_ARRIVAL': 'int16',
            'ARRIVAL_TIME': 'float32',
            'ARRIVAL_DELAY': 'float32',
            'DIVERTED': 'int8',
            'CANCELLED': 'int8',
            'CANCELLATION_REASON': 'string',
            'AIR_SYSTEM_DELAY': 'float32',
            'SECURITY_DELAY': 'float32',
            'AIRLINE_DELAY': 'float32',
            'LATE_AIRCRAFT_DELAY': 'float32',
            'WEATHER_DELAY': 'float32'
        }
        
        # Check if files exist to avoid hard crashes
        if os.path.exists('flights.parquet'):
            flights = pd.read_parquet('flights.parquet')
        elif os.path.exists('flights.csv'):
            flights = pd.read_csv('flights.csv', dtype=dtypes, low_memory=False)
        else:
            return None, None, None

        # Load reference data
        airlines = pd.read_csv('airlines.csv')
        airports = pd.read_csv('airports.csv')
        
        # Rename IATA_CODE to match flights for easier merging or lookups
        # Actually, let's keep them separate and merge on demand or just use lookups for performance
        # adding Airline Name to flights is usually helpful
        
        # Create a dictionary for fast lookups
        airline_dict = pd.Series(airlines.AIRLINE.values, index=airlines.IATA_CODE).to_dict()
        flights['AIRLINE_NAME'] = flights['AIRLINE'].map(airline_dict)

        # Optimize memory: Convert string columns to categorical
        # This reduces memory usage significantly for columns with limited unique values
        categorical_cols = ['AIRLINE', 'AIRLINE_NAME', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
                           'TAIL_NUMBER', 'CANCELLATION_REASON']
        for col in categorical_cols:
            if col in flights.columns:
                flights[col] = flights[col].astype('category')

        # Handle dates
        # Use pd.to_datetime with a dictionary to correctly handle types and avoid overflows
        flights['DATE'] = pd.to_datetime({
            'year': flights['YEAR'],
            'month': flights['MONTH'],
            'day': flights['DAY']
        })
        
        return flights, airlines, airports

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def apply_filters(df, month, airline, origin_airport):
    """
    Applies sidebar filters to the dataset.
    """
    filtered_df = df.copy() # Avoid SettingWithCopy warnings on cached data

    if month != 'All':
        # Map month names to numbers if necessary, or assume input is already correct
        # Let's assume input matches the data format (integer or mapped string)
        # If 'month' is passed as name "January", we need to map it.
        # Simple map:
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        if isinstance(month, str) and month in month_map:
             filtered_df = filtered_df[filtered_df['MONTH'] == month_map[month]]
        elif isinstance(month, int):
             filtered_df = filtered_df[filtered_df['MONTH'] == month]

    if airline != 'All':
        # filtering by AIRLINE_NAME (which we created) or AIRLINE code
        # Let's assume we filter by code for speed, or name if that's what's passed
        if len(airline) == 2: # Very rough heuristic for IATA code
             filtered_df = filtered_df[filtered_df['AIRLINE'] == airline]
        else:
             filtered_df = filtered_df[filtered_df['AIRLINE_NAME'] == airline]

    if origin_airport != 'All':
        filtered_df = filtered_df[filtered_df['ORIGIN_AIRPORT'] == origin_airport]

    return filtered_df
