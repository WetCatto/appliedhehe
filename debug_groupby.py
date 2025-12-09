import pandas as pd
try:
    print("Loading parquet...")
    df = pd.read_parquet('flights.parquet', columns=['DAY_OF_WEEK'])
    print("Loaded.")
    print("Dtype:", df['DAY_OF_WEEK'].dtype)
    
    print("Grouping...")
    counts = df.groupby('DAY_OF_WEEK').size()
    print("Counts index dtype:", counts.index.dtype)
    print("Counts per day (raw):\n", counts)
    
    print("Reindexing...")
    reindexed = counts.reindex([1, 2, 3, 4, 5, 6, 7], fill_value=0).reset_index(name='Count')
    print("Reindexed result:\n", reindexed)
    
except Exception as e:
    print(f"Error: {e}")
