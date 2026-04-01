import pandas as pd
df = pd.read_csv("data/raw/citibike/202401-citibike-tripdata_1.csv", nrows=1000)
print(df.columns)
print(df[['started_at', 'ended_at', 'start_station_name', 'end_station_name']].head())
