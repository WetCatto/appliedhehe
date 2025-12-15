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
    print(f"After removing cancelled flights: {len(ml_df):,} records")
    
    # Create target variable: 1 if delayed (>15 min), 0 otherwise
    # Fill NA values first to avoid conversion error
    ml_df['DELAYED'] = (ml_df['ARRIVAL_DELAY'].fillna(0) > 15).astype(int)
    
    # Enhanced feature engineering
    # Extract hour from scheduled departure
    ml_df['DEPARTURE_HOUR'] = (ml_df['SCHEDULED_DEPARTURE'] // 100).astype(int)
    
    # Create time of day categories
    def categorize_time(hour):
        if 5 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 17:
            return 1  # Afternoon
        elif 17 <= hour < 21:
            return 2  # Evening
        else:
            return 3  # Night/Early Morning
    
    ml_df['TIME_OF_DAY'] = ml_df['DEPARTURE_HOUR'].apply(categorize_time)
    
    # Distance categories
    ml_df['DISTANCE_CATEGORY'] = pd.cut(ml_df['DISTANCE'], 
                                         bins=[0, 500, 1500, 5000], 
                                         labels=[0, 1, 2]).astype(int)
    
    # Weekend flag
    ml_df['IS_WEEKEND'] = (ml_df['DAY_OF_WEEK'].isin([6, 7])).astype(int)
    
    # Select enhanced features
    feature_cols = [
        'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'MONTH', 'DAY_OF_WEEK', 'DAY',
        'DEPARTURE_HOUR', 'TIME_OF_DAY', 'DISTANCE', 'DISTANCE_CATEGORY',
        'IS_WEEKEND', 'TAXI_OUT'
    ]
    
    # Remove rows with missing values
    ml_df = ml_df[feature_cols + ['DELAYED']].dropna()
    print(f"After removing missing values: {len(ml_df):,} records")
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
    
    for col in categorical_cols:
        print(f"Encoding {col}...")
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        label_encoders[col] = le
    
    # Split features and target
    X = ml_df[feature_cols]
    y = ml_df['DELAYED']
    
    # Class distribution
    delayed_count = y.sum()
    ontime_count = len(y) - delayed_count
    print(f"\nClass distribution:")
    print(f"  On-time: {ontime_count:,} ({ontime_count/len(y)*100:.1f}%)")
    print(f"  Delayed: {delayed_count:,} ({delayed_count/len(y)*100:.1f}%)")
    
    return X, y, feature_cols, label_encoders

def train_model(X, y):
    """Train improved Random Forest model."""
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    print("\nTraining improved Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,          # More trees for better predictions
        max_depth=15,              # Deeper trees to capture patterns
        min_samples_split=50,      # Lower threshold for splitting
        min_samples_leaf=20,       # Lower threshold for leaves
        max_features='sqrt',       # Feature sampling for regularization
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',   # Handle class imbalance
        bootstrap=True,
        oob_score=True,           # Out-of-bag score for validation
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Training complete!")
    print(f"Out-of-bag score: {model.oob_score_:.4f}")
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
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
