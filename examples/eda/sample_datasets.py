import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random

def sample_tdrive(raw_dir="data/raw/tdrive", n_files=5, n_rows_per_file=1000):
    txt_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    sample_files = random.sample(txt_files, min(n_files, len(txt_files)))
    
    dfs = []
    for f in sample_files:
        try:
            df = pd.read_csv(f, names=['taxi_id', 'timestamp', 'longitude', 'latitude'], nrows=n_rows_per_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if dfs:
        combined = pd.concat(dfs)
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        print("\n--- T-Drive Sample ---")
        print(combined.head())
        print(f"\nShape of sample: {combined.shape}")
        return combined
    return None

def sample_citibike(raw_dir="data/raw/citibike", n_rows=5000):
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not csv_files:
       print("No CSV files found in citibike directory.")
       return None
       
    target_file = csv_files[0]
    df = pd.read_csv(target_file, nrows=n_rows)
    print("\n--- CitiBike Sample ---")
    print(df.head())
    print(f"\nShape of sample: {df.shape}")
    return df

def sample_foursquare(raw_dir="data/raw/foursquare", n_rows=5000):
    target_file = os.path.join(raw_dir, "dataset_TSMC2014_NYC.csv")
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return None
        
    df = pd.read_csv(target_file, nrows=n_rows)
    print("\n--- Foursquare Sample ---")
    print(df.head())
    print(f"\nShape of sample: {df.shape}")
    return df

if __name__ == "__main__":
    tdrive_sample = sample_tdrive()
    citibike_sample = sample_citibike()
    foursquare_sample = sample_foursquare()
