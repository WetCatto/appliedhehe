import sys
import os
# Add current dir to path to import utils
sys.path.append(os.getcwd())
try:
    from utils import load_data
    print("Testing load_data...")
    flights, airlines, airports = load_data()
    if flights is not None and not flights.empty:
        print(f"Success! Loaded {len(flights)} rows, {len(flights.columns)} columns.")
        print(f"Columns: {flights.columns.tolist()[:5]}...")
    else:
        print("Failed: Flights dataframe is None or empty.")
except Exception as e:
    print(f"Exception: {e}")
