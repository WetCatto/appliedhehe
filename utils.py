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
        
        # Create a dictionary for fast lookups
        airline_dict = pd.Series(airlines.AIRLINE.values, index=airlines.IATA_CODE).to_dict()
        flights['AIRLINE_NAME'] = flights['AIRLINE'].map(airline_dict)

        # Optimize memory: Convert string columns to categorical
        categorical_cols = ['AIRLINE', 'AIRLINE_NAME', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 
                           'TAIL_NUMBER', 'CANCELLATION_REASON']
        for col in categorical_cols:
            if col in flights.columns:
                flights[col] = flights[col].astype('category')

        # Handle dates
        flights['DATE'] = pd.to_datetime({
            'year': flights['YEAR'],
            'month': flights['MONTH'],
            'day': flights['DAY']
        })
        
        return flights, airlines, airports

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data
def compute_metrics(flights):
    """
    Pre-compute ALL dashboard metrics and aggregations once.
    This eliminates redundant computations and significantly reduces memory/CPU usage.
    Returns a dictionary containing all pre-computed data.
    """
    
    # Month and day mappings
    month_map_rev = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                     7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    day_map = {1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat', 7:'Sun'}
    
    metrics = {}
    
    # ===== SUMMARY METRICS =====
    metrics['total_flights'] = len(flights)
    metrics['on_time_flights'] = len(flights[flights['ARRIVAL_DELAY'] <= 15])
    metrics['delayed_flights'] = len(flights[flights['ARRIVAL_DELAY'] > 15])
    metrics['cancelled_flights'] = len(flights[flights['CANCELLED'] == 1])
    metrics['total_airlines'] = flights['AIRLINE'].nunique()
    metrics['total_airports'] = flights['ORIGIN_AIRPORT'].nunique()
    
    # Percentages
    metrics['on_time_pct'] = (metrics['on_time_flights'] / metrics['total_flights'] * 100)
    metrics['delayed_pct'] = (metrics['delayed_flights'] / metrics['total_flights'] * 100)
    metrics['cancelled_pct'] = (metrics['cancelled_flights'] / metrics['total_flights'] * 100)
    
    # ===== DELAY TAB METRICS =====
    metrics['avg_airline_delay'] = flights['AIRLINE_DELAY'].mean()
    metrics['avg_aircraft_delay'] = flights['LATE_AIRCRAFT_DELAY'].mean()
    metrics['avg_system_delay'] = flights['AIR_SYSTEM_DELAY'].mean()
    metrics['avg_weather_delay'] = flights['WEATHER_DELAY'].mean()
    metrics['avg_security_delay'] = flights['SECURITY_DELAY'].mean()
    
    # Delay by month
    delay_means_month = flights.groupby('MONTH').agg({
        'AIRLINE_DELAY': 'mean',
        'LATE_AIRCRAFT_DELAY': 'mean',
        'AIR_SYSTEM_DELAY': 'mean',
        'WEATHER_DELAY': 'mean',
        'DEPARTURE_DELAY': 'mean',
        'ARRIVAL_DELAY': 'mean',
        'TAXI_OUT': 'mean'
    }).reset_index()
    delay_means_month['Avg Airline & Aircraft Delay'] = delay_means_month['AIRLINE_DELAY'] + delay_means_month['LATE_AIRCRAFT_DELAY']
    delay_means_month['Avg Air System Delay'] = delay_means_month['AIR_SYSTEM_DELAY']
    delay_means_month['Month'] = delay_means_month['MONTH'].map(month_map_rev)
    metrics['delay_means_month'] = delay_means_month
    
    # Average delay by airline and month
    avg_delay_airline_month = flights.groupby(['MONTH', 'AIRLINE_NAME'])['ARRIVAL_DELAY'].mean().reset_index()
    avg_delay_airline_month['Month'] = avg_delay_airline_month['MONTH'].map(month_map_rev)
    metrics['avg_delay_airline_month'] = avg_delay_airline_month
    
    # ===== TIME TAB METRICS =====
    # Month stats
    month_stats = flights.groupby('MONTH').apply(lambda x: pd.Series({
        'Total': len(x),
        'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
        'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
        'Cancelled': x['CANCELLED'].sum()
    }), include_groups=False).reset_index()
    month_stats['Month Name'] = month_stats['MONTH'].map(month_map_rev)
    metrics['month_stats'] = month_stats
    
    # DOW stats
    dow_stats = flights.groupby('DAY_OF_WEEK').apply(lambda x: pd.Series({
        'Total': len(x),
        'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
        'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
        'Cancelled': x['CANCELLED'].sum()
    }), include_groups=False).reset_index()
    dow_stats['Day Name'] = dow_stats['DAY_OF_WEEK'].map(day_map)
    dow_stats = dow_stats.sort_values('Total', ascending=False)
    metrics['dow_stats'] = dow_stats
    
    # Day stats (1-31)
    day_stats = flights.groupby('DAY').apply(lambda x: pd.Series({
        'Total': len(x),
        'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
        'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
        'Cancelled': x['CANCELLED'].sum()
    }), include_groups=False).reset_index()
    metrics['day_stats'] = day_stats
    
    # Delay type by month
    delay_stream = flights.groupby('MONTH')[['AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'AIR_SYSTEM_DELAY', 
                                              'SECURITY_DELAY', 'WEATHER_DELAY', 'TAXI_OUT']].sum().reset_index()
    delay_stream['Month'] = delay_stream['MONTH'].map(month_map_rev)
    metrics['delay_stream'] = delay_stream
    
    # ===== AIRLINE TAB METRICS =====
    aa_delay_df = flights[(flights['AIRLINE_DELAY'] > 0) | (flights['LATE_AIRCRAFT_DELAY'] > 0)]
    metrics['aa_delay_count'] = len(aa_delay_df)
    metrics['aa_delay_pct'] = (metrics['aa_delay_count'] / metrics['total_flights'] * 100)
    metrics['avg_dep'] = flights['DEPARTURE_DELAY'].mean()
    metrics['avg_arr'] = flights['ARRIVAL_DELAY'].mean()
    metrics['avg_aa_delay'] = (flights['AIRLINE_DELAY'].fillna(0) + flights['LATE_AIRCRAFT_DELAY'].fillna(0)).mean()
    
    # Airline counts
    airline_counts = flights.groupby('AIRLINE_NAME', observed=False).apply(lambda x: pd.Series({
        'Total': len(x),
        'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
        'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
        'Cancelled': x['CANCELLED'].sum()
    }), include_groups=False).reset_index()
    airline_counts = airline_counts.sort_values('Total', ascending=False)
    airline_counts['On Time %'] = (airline_counts['On Time'] / airline_counts['Total'] * 100).round(2)
    airline_counts['Delayed %'] = (airline_counts['Delayed'] / airline_counts['Total'] * 100).round(2)
    airline_counts['Cancelled %'] = (airline_counts['Cancelled'] / airline_counts['Total'] * 100).round(2)
    metrics['airline_counts'] = airline_counts
    
    # ===== AIRPORT TAB METRICS =====
    # Top 50 airports
    top_airports_list = flights['ORIGIN_AIRPORT'].value_counts().nlargest(50).index.tolist()
    airport_df_top = flights[flights['ORIGIN_AIRPORT'].isin(top_airports_list)]
    
    airport_counts = airport_df_top.groupby('ORIGIN_AIRPORT', observed=False).apply(lambda x: pd.Series({
        'Total': len(x),
        'On Time': ((x['ARRIVAL_DELAY'] <= 15) & (x['CANCELLED'] == 0)).sum(),
        'Delayed': (x['ARRIVAL_DELAY'] > 15).sum(),
        'Cancelled': x['CANCELLED'].sum()
    }), include_groups=False).reset_index()
    airport_counts['On Time %'] = (airport_counts['On Time'] / airport_counts['Total'] * 100).fillna(0)
    airport_counts['Delayed %'] = (airport_counts['Delayed'] / airport_counts['Total'] * 100).fillna(0)
    airport_counts['Cancelled %'] = (airport_counts['Cancelled'] / airport_counts['Total'] * 100).fillna(0)
    metrics['airport_counts'] = airport_counts
    
    # ===== EDA TAB METRICS =====
    # Volume by month
    vol_month = flights.groupby('MONTH').size().reset_index(name='Count')
    vol_month['Month Name'] = vol_month['MONTH'].map(month_map_rev)
    metrics['vol_month'] = vol_month
    
    # Volume by day of week
    vol_day = flights.groupby('DAY_OF_WEEK').size().reindex([1, 2, 3, 4, 5, 6, 7], fill_value=0).reset_index(name='Count')
    vol_day['Day Name'] = vol_day['DAY_OF_WEEK'].map(day_map)
    metrics['vol_day'] = vol_day
    
    # Departure delay insights
    dep_delayed_flights = flights[flights['DEPARTURE_DELAY'] > 15]
    metrics['pct_dep_delayed'] = (len(dep_delayed_flights) / len(flights) * 100)
    metrics['avg_dep_delay_duration'] = dep_delayed_flights['DEPARTURE_DELAY'].mean() if len(dep_delayed_flights) > 0 else 0
    
    # Delay trends
    monthly_stats = flights.groupby('MONTH')['DEPARTURE_DELAY'].apply(lambda x: (x > 15).mean() * 100).reset_index(name='Delayed_Pct')
    monthly_stats['Type'] = 'National Average'
    monthly_stats['Month'] = monthly_stats['MONTH'].map(month_map_rev)
    
    bos_flights = flights[flights['ORIGIN_AIRPORT'] == 'BOS']
    if len(bos_flights) > 0:
        bos_stats = bos_flights.groupby('MONTH')['DEPARTURE_DELAY'].apply(lambda x: (x > 15).mean() * 100).reset_index(name='Delayed_Pct')
        bos_stats['Type'] = 'Boston (BOS)'
        bos_stats['Month'] = bos_stats['MONTH'].map(month_map_rev)
        comp_trend = pd.concat([monthly_stats, bos_stats])
    else:
        comp_trend = monthly_stats
    metrics['comp_trend'] = comp_trend
    
    # Cancellation analysis
    metrics['total_cancelled'] = flights['CANCELLED'].sum()
    metrics['pct_cancelled'] = (metrics['total_cancelled'] / len(flights) * 100)
    
    cancel_counts = flights[flights['CANCELLED'] == 1]['CANCELLATION_REASON'].value_counts()
    val_A = cancel_counts.get('A', 0)
    val_B = cancel_counts.get('B', 0)
    metrics['pct_weather'] = (val_B / metrics['total_cancelled'] * 100) if metrics['total_cancelled'] > 0 else 0
    metrics['pct_airline'] = (val_A / metrics['total_cancelled'] * 100) if metrics['total_cancelled'] > 0 else 0
    
    return metrics
