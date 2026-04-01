import pandas as pd
import numpy as np

print("="*60)
print("EDA: T-DRIVE DATASET (Routing/Navigation)")
print("="*60)
try:
    tdrive = pd.read_csv("data/samples/tdrive_sample.csv", parse_dates=['timestamp'])
    print(f"Shape: {tdrive.shape}\n")
    print("--- Missing Values ---")
    print(tdrive.isna().sum())
    print("\n--- Unique Values ---")
    print(f"{tdrive['taxi_id'].nunique()} Unique Taxis")
    print("\n--- Trajectory Characteristics ---")
    print(f"Time span: {tdrive['timestamp'].min()} to {tdrive['timestamp'].max()}")
    points_per_taxi = tdrive.groupby('taxi_id').size()
    print(f"Median points per taxi: {points_per_taxi.median()}")
    print(f"Max points per taxi: {points_per_taxi.max()}")
except Exception as e:
    print(e)

print("\n" + "="*60)
print("EDA: CITIBIKE DATASET (Gig-Economy/Labor Supply)")
print("="*60)
try:
    citibike = pd.read_csv("data/samples/citibike_sample.csv", parse_dates=['started_at', 'ended_at'])
    print(f"Shape: {citibike.shape}\n")
    print("--- Missing Values ---")
    missing = citibike.isna().sum()
    print(missing[missing > 0]) # only showing columns with missing
    print("\n--- Unique Values ---")
    print(f"Start Stations: {citibike['start_station_name'].nunique()}")
    print(f"End Stations: {citibike['end_station_name'].nunique()}")
    print("\n--- Trip Characteristics ---")
    durations = (citibike['ended_at'] - citibike['started_at']).dt.total_seconds() / 60
    print(f"Median Trip Duration (mins): {durations.median():.2f}")
    print("\nTop 3 Start Stations:")
    print(citibike['start_station_name'].value_counts().head(3))
except Exception as e:
    print(e)

print("\n" + "="*60)
print("EDA: FOURSQUARE DATASET (Sequential Recommendation)")
print("="*60)
try:
    foursquare = pd.read_csv("data/samples/foursquare_sample.csv")
    print(f"Shape: {foursquare.shape}\n")
    print("--- Missing Values ---")
    print(foursquare.isna().sum())
    print("\n--- Unique Values ---")
    print(f"Users: {foursquare['userId'].nunique()}, Venues: {foursquare['venueId'].nunique()}, Categories: {foursquare['venueCategory'].nunique()}")
    print("\n--- User Characteristics ---")
    checkins_per_user = foursquare.groupby('userId').size()
    print(f"Median Check-ins per User: {checkins_per_user.median()}")
    print("\nTop 5 Venue Categories:")
    print(foursquare['venueCategory'].value_counts().head(5))
except Exception as e:
    print(e)
