import pandas as pd
import streamlit as st
import os

@st.cache_data(ttl=3600, max_entries=1)
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
    
    # helper columns for vectorized aggregation (much faster & less memory than applying per group)
    # Use 1/0 integers for counting
    flights['is_on_time'] = ((flights['ARRIVAL_DELAY'] <= 15) & (flights['CANCELLED'] == 0)).fillna(False).astype(int)
    flights['is_delayed'] = (flights['ARRIVAL_DELAY'] > 15).fillna(False).astype(int)
    # flights['CANCELLED'] is already int
    
    metrics['on_time_flights'] = flights['is_on_time'].sum()
    metrics['delayed_flights'] = flights['is_delayed'].sum()
    metrics['cancelled_flights'] = flights['CANCELLED'].sum()
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
    avg_delay_airline_month = flights.groupby(['MONTH', 'AIRLINE_NAME'], observed=False)['ARRIVAL_DELAY'].mean().reset_index()
    avg_delay_airline_month['Month'] = avg_delay_airline_month['MONTH'].map(month_map_rev)
    metrics['avg_delay_airline_month'] = avg_delay_airline_month
    
    # ===== TIME TAB METRICS =====
    # Month stats
    # Month stats
    # OPTIMIZATION: Use vectorized agg instead of apply
    month_stats = flights.groupby('MONTH', observed=False).agg(
        Total=('MONTH', 'size'),
        On_Time=('is_on_time', 'sum'),
        Delayed=('is_delayed', 'sum'),
        Cancelled=('CANCELLED', 'sum')
    ).reset_index()
    # Rename for compatibility
    month_stats = month_stats.rename(columns={'On_Time': 'On Time', 'Delayed': 'Delayed', 'Cancelled': 'Cancelled'})
    month_stats['Month Name'] = month_stats['MONTH'].map(month_map_rev)
    metrics['month_stats'] = month_stats
    
    # DOW stats
    # DOW stats
    dow_stats = flights.groupby('DAY_OF_WEEK', observed=False).agg(
        Total=('DAY_OF_WEEK', 'size'),
        On_Time=('is_on_time', 'sum'),
        Delayed=('is_delayed', 'sum'),
        Cancelled=('CANCELLED', 'sum')
    ).reset_index()
    dow_stats = dow_stats.rename(columns={'On_Time': 'On Time', 'Delayed': 'Delayed', 'Cancelled': 'Cancelled'})
    dow_stats['Day Name'] = dow_stats['DAY_OF_WEEK'].map(day_map)
    dow_stats = dow_stats.sort_values('DAY_OF_WEEK', ascending=True)
    metrics['dow_stats'] = dow_stats
    
    # Day stats (1-31)
    # Day stats (1-31)
    day_stats = flights.groupby('DAY', observed=False).agg(
        Total=('DAY', 'size'),
        On_Time=('is_on_time', 'sum'),
        Delayed=('is_delayed', 'sum'),
        Cancelled=('CANCELLED', 'sum')
    ).reset_index()
    day_stats = day_stats.rename(columns={'On_Time': 'On Time', 'Delayed': 'Delayed', 'Cancelled': 'Cancelled'})
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
    # Airline counts
    airline_counts = flights.groupby('AIRLINE_NAME', observed=False).agg(
        Total=('AIRLINE_NAME', 'size'),
        On_Time=('is_on_time', 'sum'),
        Delayed=('is_delayed', 'sum'),
        Cancelled=('CANCELLED', 'sum')
    ).reset_index()
    airline_counts = airline_counts.rename(columns={'On_Time': 'On Time', 'Delayed': 'Delayed', 'Cancelled': 'Cancelled'})
    airline_counts = airline_counts.sort_values('Total', ascending=False)
    airline_counts['On Time %'] = (airline_counts['On Time'] / airline_counts['Total'] * 100).round(2)
    airline_counts['Delayed %'] = (airline_counts['Delayed'] / airline_counts['Total'] * 100).round(2)
    airline_counts['Cancelled %'] = (airline_counts['Cancelled'] / airline_counts['Total'] * 100).round(2)
    metrics['airline_counts'] = airline_counts
    
    # ===== AIRPORT TAB METRICS =====
    # Top 50 airports
    top_airports_list = flights['ORIGIN_AIRPORT'].value_counts().nlargest(50).index.tolist()
    airport_df_top = flights[flights['ORIGIN_AIRPORT'].isin(top_airports_list)]
    
    airport_counts = airport_df_top.groupby('ORIGIN_AIRPORT', observed=False).agg(
        Total=('ORIGIN_AIRPORT', 'size'),
        On_Time=('is_on_time', 'sum'),
        Delayed=('is_delayed', 'sum'),
        Cancelled=('CANCELLED', 'sum')
    ).reset_index()
    airport_counts = airport_counts.rename(columns={'On_Time': 'On Time', 'Delayed': 'Delayed', 'Cancelled': 'Cancelled'})
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


# ===== MACHINE LEARNING FUNCTIONS =====

@st.cache_data
def prepare_ml_data(_flights):
    """
    Prepare data for machine learning model with improved features.
    Returns: X_train, X_test, y_train, y_test, feature_names, label_encoders
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    # Create a copy and remove cancelled flights
    ml_df = _flights[_flights['CANCELLED'] == 0].copy()
    
    # Create target variable: 1 if delayed (>15 min), 0 otherwise
    ml_df['DELAYED'] = (ml_df['ARRIVAL_DELAY'].fillna(0) > 15).astype(int)
    
    # --- TARGET ENCODING ---
    # Calculate risk scores on FULL dataset
    airline_risk = ml_df.groupby('AIRLINE', observed=False)['DELAYED'].mean()
    ml_df['AIRLINE_RISK'] = ml_df['AIRLINE'].map(airline_risk)
    
    origin_risk = ml_df.groupby('ORIGIN_AIRPORT', observed=False)['DELAYED'].mean()
    ml_df['ORIGIN_RISK'] = ml_df['ORIGIN_AIRPORT'].map(origin_risk)
    
    dest_risk = ml_df.groupby('DESTINATION_AIRPORT', observed=False)['DELAYED'].mean()
    ml_df['DEST_RISK'] = ml_df['DESTINATION_AIRPORT'].map(dest_risk)
    
    target_encoders = {
        'AIRLINE_RISK': airline_risk.to_dict(),
        'ORIGIN_RISK': origin_risk.to_dict(),
        'DEST_RISK': dest_risk.to_dict()
    }

    # --- BALANCED DOWNSAMPLING FOR MAX F1 SCORE ---
    TARGET_SIZE_PER_CLASS = 125000
    
    df_delayed = ml_df[ml_df['DELAYED'] == 1]
    df_ontime = ml_df[ml_df['DELAYED'] == 0]
    
    if len(df_delayed) >= TARGET_SIZE_PER_CLASS and len(df_ontime) >= TARGET_SIZE_PER_CLASS:
        df_delayed = df_delayed.sample(n=TARGET_SIZE_PER_CLASS, random_state=42)
        df_ontime = df_ontime.sample(n=TARGET_SIZE_PER_CLASS, random_state=42)
        ml_df = pd.concat([df_delayed, df_ontime])
        # Shuffle
        ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Fallback
        ml_df = ml_df.sample(n=TARGET_SIZE_PER_CLASS*2, random_state=42)
    
    # Enhanced feature engineering
    ml_df['DEPARTURE_HOUR'] = (ml_df['SCHEDULED_DEPARTURE'] // 100).astype(int)
    
    def categorize_time(hour):
        if 5 <= hour < 12: return 0  # Morning
        elif 12 <= hour < 17: return 1  # Afternoon
        elif 17 <= hour < 21: return 2  # Evening
        else: return 3  # Night
    
    ml_df['TIME_OF_DAY'] = ml_df['DEPARTURE_HOUR'].apply(categorize_time)
    
    ml_df['DISTANCE_CATEGORY'] = pd.cut(ml_df['DISTANCE'], bins=[0, 500, 1500, 5000], labels=[0, 1, 2]).astype(int)
    ml_df['IS_WEEKEND'] = (ml_df['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    
    # Select enhanced features
    feature_cols = [
        'AIRLINE_RISK', 'ORIGIN_RISK', 'DEST_RISK',
        'MONTH', 'DAY_OF_WEEK', 'DAY',
        'DEPARTURE_HOUR', 'TIME_OF_DAY', 'DISTANCE', 'DISTANCE_CATEGORY',
        'IS_WEEKEND', 'TAXI_OUT'
    ]
    
    # Remove rows with missing values
    ml_df = ml_df[feature_cols + ['DELAYED']].dropna()
    
    X = ml_df[feature_cols]
    y = ml_df['DELAYED']
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols, target_encoders


@st.cache_resource
def train_delay_model(_X_train, _y_train):
    """
    Train improved Random Forest model for delay prediction.
    Uses @st.cache_resource to avoid retraining on every reload.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Train Random Forest with improved parameters for better performance
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        # class_weight='balanced', # REMOVED: Data is already 50/50 corrected
        bootstrap=True,
        oob_score=True
    )
    
    model.fit(_X_train, _y_train)
    return model


@st.cache_data
def get_model_metrics(_model, _X_test, _y_test):
    """
    Calculate model evaluation metrics.
    Returns: dict with accuracy, precision, recall, f1, confusion_matrix, roc_auc
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    y_pred = _model.predict(_X_test)
    y_prob = _model.predict_proba(_X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(_y_test, y_pred),
        'precision': precision_score(_y_test, y_pred),
        'recall': recall_score(_y_test, y_pred),
        'f1': f1_score(_y_test, y_pred),
        'roc_auc': roc_auc_score(_y_test, y_prob),
        'confusion_matrix': confusion_matrix(_y_test, y_pred)
    }
    
    return metrics


def predict_delay_probability(model, target_encoders, airline, origin, destination, 
                              month, day_of_week, day, scheduled_dep, distance, taxi_out=10):
    """
    Predict delay probability using Target Encoding risks.
    Returns: probability of delay (0-1)
    """
    import numpy as np
    
    # Look up Risk Scores from target_encoders
    # Default to global average (~0.18) if unknown
    GLOBAL_AVG_DELAY = 0.18
    
    airline_risk = target_encoders['AIRLINE_RISK'].get(airline, GLOBAL_AVG_DELAY)
    origin_risk = target_encoders['ORIGIN_RISK'].get(origin, GLOBAL_AVG_DELAY)
    dest_risk = target_encoders['DEST_RISK'].get(destination, GLOBAL_AVG_DELAY)
    
    # Extract hour from scheduled departure
    departure_hour = scheduled_dep // 100
    
    # Time of day category
    if 5 <= departure_hour < 12: time_of_day = 0  # Morning
    elif 12 <= departure_hour < 17: time_of_day = 1  # Afternoon
    elif 17 <= departure_hour < 21: time_of_day = 2  # Evening
    else: time_of_day = 3  # Night
    
    # Distance category
    if distance <= 500: distance_category = 0
    elif distance <= 1500: distance_category = 1
    else: distance_category = 2
    
    # Weekend flag
    is_weekend = 1 if day_of_week in [6, 7] else 0
    
    # Create feature array matching training columns
    # Order: AIRLINE_RISK, ORIGIN_RISK, DEST_RISK, MONTH, DAY_OF_WEEK, DAY,
    #        DEPARTURE_HOUR, TIME_OF_DAY, DISTANCE, DISTANCE_CATEGORY, IS_WEEKEND, TAXI_OUT
    # Create DataFrame matching training columns to avoid warnings
    # Order: AIRLINE_RISK, ORIGIN_RISK, DEST_RISK, MONTH, DAY_OF_WEEK, DAY,
    #        DEPARTURE_HOUR, TIME_OF_DAY, DISTANCE, DISTANCE_CATEGORY, IS_WEEKEND, TAXI_OUT
    features = pd.DataFrame([[
        airline_risk, origin_risk, dest_risk,
        month, day_of_week, day,
        departure_hour, time_of_day, distance, distance_category,
        is_weekend, taxi_out
    ]], columns=[
        'AIRLINE_RISK', 'ORIGIN_RISK', 'DEST_RISK',
        'MONTH', 'DAY_OF_WEEK', 'DAY',
        'DEPARTURE_HOUR', 'TIME_OF_DAY', 'DISTANCE', 'DISTANCE_CATEGORY',
        'IS_WEEKEND', 'TAXI_OUT'
    ])
    
    # Predict probability
    prob = model.predict_proba(features)[0][1] 
    
    return prob


@st.cache_resource
def load_trained_model_v2():
    """
    Load pre-trained model from disk.
    If model file doesn't exist, returns None and app will train on-the-fly.
    
    Returns: dict with model, feature_names, label_encoders, metrics (or None)
    """
    import pickle
    import os
    
    model_path = os.path.join(os.path.dirname(__file__), 'flight_delay_model.pkl')
    
    # Print to logs to verify loading (and bust cache)
    print("Loading optimized flight delay model...")
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            st.warning(f"Could not load pre-trained model: {e}")
            return None
    else:
        return None
