import pandas as pd
import numpy as np
import pyarrow.parquet as pq

print("============================================================")
print("EDA: NYC TLC YELLOW TAXI DATA (Jan 2024)")
print("============================================================")
try:
    taxi_file = "/Volumes/Expansion/datasets/nyc_tlc/yellow_tripdata_2024-01.parquet"
    # Just read the first 10,000 rows for EDA to save memory
    taxi_table = pq.read_table(taxi_file)
    taxi_df = taxi_table.slice(0, 10000).to_pandas()
    print(f"Total rows in dataset: {taxi_table.num_rows}")
    print(f"Columns: {list(taxi_df.columns)}")
    print(f"\nEarnings Sample Summary:\n{taxi_df[['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']].describe().loc[['mean', 'min', 'max']]}")
except Exception as e:
    print(f"Error: {e}")

print("\n============================================================")
print("EDA: NYC TLC HVFHV (UBER/LYFT) DATA (Jan 2024)")
print("============================================================")
try:
    uber_file = "/Volumes/Expansion/datasets/nyc_tlc/fhvhv_tripdata_2024-01.parquet"
    uber_table = pq.read_table(uber_file)
    uber_df = uber_table.slice(0, 10000).to_pandas()
    print(f"Total rows in dataset: {uber_table.num_rows}")
    print(f"Columns: {list(uber_df.columns)}")
    print(f"\nEarnings Sample Summary:\n{uber_df[['trip_miles', 'trip_time', 'base_passenger_fare', 'driver_pay']].describe().loc[['mean', 'min', 'max']]}")
except Exception as e:
    print(f"Error: {e}")

print("\n============================================================")
print("EDA: NYC WEATHER COVARIATES (Jan 2024)")
print("============================================================")
try:
    weather_file = "/Volumes/Expansion/datasets/weather/nyc_weather_jan2024.csv"
    weather_df = pd.read_csv(weather_file)
    print(f"Total hours recorded: {len(weather_df)}")
    print(f"Columns: {list(weather_df.columns)}")
    print(f"\nWeather Summary:\n{weather_df.describe().loc[['mean', 'min', 'max']]}")
except Exception as e:
    print(f"Error: {e}")
