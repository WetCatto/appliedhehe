"""
Train the flight delay prediction model and save it to disk.

This script should be run once to train the model. The trained model
will be saved as 'flight_delay_model.pkl' and can be loaded by the
Streamlit app for predictions.

Usage:
    python train_model.py
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def load_flight_data():
    """Load and prepare flight data."""
    print("Loading flight data...")
    
    if os.path.exists('flights.parquet'):
        flights = pd.read_parquet('flights.parquet')
    elif os.path.exists('flights.csv'):
        flights = pd.read_csv('flights.csv', low_memory=False)
    else:
        raise FileNotFoundError("No flight data file found (flights.parquet or flights.csv)")
    
    print(f"Loaded {len(flights):,} flight records")
    return flights

def prepare_data(flights):
    """Prepare data for machine learning with improved features."""
    print("Preparing data for ML...")
    import numpy as np
    
    # Remove cancelled flights
    ml_df = flights[flights['CANCELLED'] == 0].copy()
    print(f"Total records available: {len(ml_df):,}")

    # Create target variable: 1 if delayed (>15 min), 0 otherwise
    ml_df['DELAYED'] = (ml_df['ARRIVAL_DELAY'].fillna(0) > 15).astype(int)

    # --- TARGET ENCODING (The "Secret Sauce") ---
    # Calculate historical delay rates on the FULL dataset before downsampling
    # This gives us highly accurate "Risk Scores" based on 5M+ flights
    print("Calculating historical risk scores on full dataset...")
    
    # 1. Airline Risk
    airline_risk = ml_df.groupby('AIRLINE', observed=False)['DELAYED'].mean()
    ml_df['AIRLINE_RISK'] = ml_df['AIRLINE'].map(airline_risk)
    
    # 2. Origin Airport Risk
    origin_risk = ml_df.groupby('ORIGIN_AIRPORT', observed=False)['DELAYED'].mean()
    ml_df['ORIGIN_RISK'] = ml_df['ORIGIN_AIRPORT'].map(origin_risk)
    
    # 3. Destination Airport Risk
    dest_risk = ml_df.groupby('DESTINATION_AIRPORT', observed=False)['DELAYED'].mean()
    ml_df['DEST_RISK'] = ml_df['DESTINATION_AIRPORT'].map(dest_risk)
    
    # Save these mappings to use for prediction later
    target_encoders = {
        'AIRLINE_RISK': airline_risk.to_dict(),
        'ORIGIN_RISK': origin_risk.to_dict(),
        'DEST_RISK': dest_risk.to_dict()
    }

    # --- BALANCED DOWNSAMPLING FOR MAX F1 SCORE ---
    # To get high F1, we need to show the model equal examples of both classes.
    # We will create a 50/50 dataset: 125k Delayed + 125k On-Time.
    TARGET_SIZE_PER_CLASS = 125000
    
    df_delayed = ml_df[ml_df['DELAYED'] == 1]
    df_ontime = ml_df[ml_df['DELAYED'] == 0]
    
    print(f"Available Delays: {len(df_delayed):,}")
    print(f"Available On-Time: {len(df_ontime):,}")
    
    if len(df_delayed) >= TARGET_SIZE_PER_CLASS and len(df_ontime) >= TARGET_SIZE_PER_CLASS:
        print(f"Creating balanced dataset: {TARGET_SIZE_PER_CLASS:,} of each class...")
        df_delayed = df_delayed.sample(n=TARGET_SIZE_PER_CLASS, random_state=42)
        df_ontime = df_ontime.sample(n=TARGET_SIZE_PER_CLASS, random_state=42)
        ml_df = pd.concat([df_delayed, df_ontime])
        # Shuffle
        ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Fallback if not enough data (unlikely)
        print("Not enough data for balanced target, using random sample...")
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
    
    # Select features - replacing raw IDs with Risk Scores
    feature_cols = [
        'AIRLINE_RISK', 'ORIGIN_RISK', 'DEST_RISK', # New powerful features
        'MONTH', 'DAY_OF_WEEK', 'DAY',
        'DEPARTURE_HOUR', 'TIME_OF_DAY', 'DISTANCE', 'DISTANCE_CATEGORY',
        'IS_WEEKEND', 'TAXI_OUT'
    ]
    
    # Remove rows with missing values
    ml_df = ml_df[feature_cols + ['DELAYED']].dropna()
    print(f"Training data ready: {len(ml_df):,} records")
    
    X = ml_df[feature_cols]
    y = ml_df['DELAYED']
    
    # Class distribution
    delayed_count = y.sum()
    ontime_count = len(y) - delayed_count
    print(f"Class distribution: On-time: {ontime_count:,} ({ontime_count/len(y)*100:.1f}%) | Delayed: {delayed_count:,} ({delayed_count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols, target_encoders

def train_model(X, y):
    """Train improved Random Forest model."""
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    print("\nTraining improved Random Forest model...")
    # Balanced data = No need for class_weight
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
        oob_score=True,           
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    print(f"Out-of-bag score: {model.oob_score_:.4f}")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("\nEvaluating model...")
    from sklearn.metrics import roc_auc_score
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]:,}")
    print(f"  False Positives: {cm[0][1]:,}")
    print(f"  False Negatives: {cm[1][0]:,}")
    print(f"  True Positives:  {cm[1][1]:,}")
    print("="*50)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics

def save_model(model, feature_names, label_encoders, metrics):
    """Save model and associated data to disk."""
    print("\nSaving model to disk...")
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'metrics': metrics
    }
    
    with open('flight_delay_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    file_size = os.path.getsize('flight_delay_model.pkl') / (1024 * 1024)
    print(f"Model saved to 'flight_delay_model.pkl' ({file_size:.2f} MB)")

def main():
    """Main training pipeline."""
    print("\n" + "="*50)
    print("FLIGHT DELAY PREDICTION MODEL TRAINING")
    print("="*50 + "\n")
    
    # Load data
    flights = load_flight_data()
    
    # Prepare data
    X, y, feature_names, label_encoders = prepare_data(flights)
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, feature_names, label_encoders, metrics)
    
    print("\n✅ Training pipeline complete!")
    print("The model is ready to use in the Streamlit app.\n")

if __name__ == "__main__":
    main()
