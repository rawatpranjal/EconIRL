import pandas as pd
import numpy as np
import os
import glob
import random

OUT_DIR = "/Volumes/Expansion/datasets/econirl_samples/"

# 1. T-Drive: Sample 500 taxis (files) completely
tdrive_files = glob.glob("data/raw/tdrive/*.txt")
if tdrive_files:
    sample_files = random.sample(tdrive_files, min(500, len(tdrive_files)))
    dfs = []
    for f in sample_files:
        try:
            df = pd.read_csv(f, names=['taxi_id', 'timestamp', 'longitude', 'latitude'])
            dfs.append(df)
        except Exception as e:
            pass
    if dfs:
        combined = pd.concat(dfs)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        out_path = os.path.join(OUT_DIR, "tdrive_medium.csv")
        combined.to_csv(out_path, index=False)
        print(f"Saved T-Drive ({combined.shape[0]} rows) to {out_path}")

# 2. CitiBike: Sample 500,000 rows
citibike_files = glob.glob("data/raw/citibike/*.csv")
if citibike_files:
    try:
        df = pd.read_csv(citibike_files[0], nrows=500000)
        out_path = os.path.join(OUT_DIR, "citibike_medium.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved CitiBike ({df.shape[0]} rows) to {out_path}")
    except Exception as e:
        print(f"Error saving CitiBike: {e}")

# 3. Foursquare: Save the entire NYC dataset (it's only 28MB, ~227k rows)
fs_file = "data/raw/foursquare/dataset_TSMC2014_NYC.csv"
if os.path.exists(fs_file):
    try:
        df = pd.read_csv(fs_file) # The full file is a great medium size
        out_path = os.path.join(OUT_DIR, "foursquare_medium.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved Foursquare ({df.shape[0]} rows) to {out_path}")
    except Exception as e:
         print(f"Error saving Foursquare: {e}")
