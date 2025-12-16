import pandas as pd
try:
    df = pd.read_parquet('data/processed/flights.parquet', columns=['DAY_OF_WEEK'])
    print("Unique DAY_OF_WEEK values:", sorted(df['DAY_OF_WEEK'].unique().tolist()))
    print("Value Counts:\n", df['DAY_OF_WEEK'].value_counts().sort_index())
except Exception as e:
    print(e)
