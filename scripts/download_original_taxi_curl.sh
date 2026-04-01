#!/bin/bash

out_dir="/Volumes/Expansion/datasets/nyc_yellow_taxi_2013_original"
mkdir -p "$out_dir"

echo "Downloading Trip Data (using correct casing and identifier)..."
curl -L 'https://archive.org/download/nycTaxiTripData2013/trip_data.7z/trip_data_1.csv' -o "$out_dir/trip_data_1.csv"
