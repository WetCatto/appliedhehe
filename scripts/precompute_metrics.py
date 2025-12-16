#!/usr/bin/env python3
"""
Precompute Dashboard Metrics Script

This script precomputes all dashboard metrics using the full flights dataset
and saves them to disk. This allows Streamlit Cloud to load precomputed metrics
instead of computing them on-the-fly, drastically reducing memory usage.

Usage:
    python precompute_metrics.py

Output:
    precomputed_metrics.pkl - Commit this file to your repository
"""

import pandas as pd
import joblib
import os
import sys
from datetime import datetime

def main():
    print("=" * 60)
    print("Flight Dashboard Metrics Precomputation")
    print("=" * 60)
    print()
    
    # Import the compute_metrics function from utils
    try:
        from utils import load_data, compute_metrics
    except ImportError as e:
        print(f"❌ Error: Could not import utils module: {e}")
        print("Make sure you're running this script from the project directory.")
        sys.exit(1)
    
    # Load full dataset
    print("📊 Loading full dataset...")
    flights, airlines, airports = load_data()
    
    if flights is None or flights.empty:
        print("❌ Error: Could not load flights data")
        print("Make sure data/processed/flights.parquet exists in the current directory")
        sys.exit(1)
    
    print(f"✅ Loaded {len(flights):,} flight records")
    print(f"✅ Loaded {len(airlines)} airlines")
    print(f"✅ Loaded {len(airports)} airports")
    print()
    
    # Compute all metrics
    print("🔄 Computing dashboard metrics (this may take a minute)...")
    metrics = compute_metrics(flights)
    
    # Add flag to indicate these are precomputed metrics
    metrics['_precomputed'] = True
    
    print("✅ Metrics computed successfully")
    print()
    
    # Display summary of computed metrics
    print("📋 Computed Metrics Summary:")
    print("-" * 60)
    print(f"  Total Flights:     {metrics['total_flights']:,}")
    print(f"  On-Time Flights:   {metrics['on_time_flights']:,} ({metrics['on_time_pct']:.1f}%)")
    print(f"  Delayed Flights:   {metrics['delayed_flights']:,} ({metrics['delayed_pct']:.1f}%)")
    print(f"  Cancelled Flights: {metrics['cancelled_flights']:,} ({metrics['cancelled_pct']:.1f}%)")
    print(f"  Total Airlines:    {metrics['total_airlines']}")
    print(f"  Total Airports:    {metrics['total_airports']}")
    print("-" * 60)
    print()
    
    # Save to disk
    output_file = 'data/processed/precomputed_metrics.pkl'
    print(f"💾 Saving metrics to {output_file}...")
    
    try:
        joblib.dump(metrics, output_file, compress=3)
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Saved successfully ({file_size:.2f} MB)")
        print()
        
        # Add metadata
        print("📝 Metadata:")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  File size: {file_size:.2f} MB")
        print()
        
        # Instructions
        print("=" * 60)
        print("✅ PRECOMPUTATION COMPLETE")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Commit data/processed/precomputed_metrics.pkl to your repository:")
        print(f"     git add {output_file}")
        print(f"     git commit -m 'Add precomputed metrics for cloud deployment'")
        print()
        print("  2. Update your app to use precomputed metrics (if not done yet)")
        print()
        print("  3. Deploy to Streamlit Cloud")
        print()
        print("To refresh metrics with updated data, re-run this script.")
        print()
        
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
