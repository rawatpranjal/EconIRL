#!/bin/bash
mkdir -p /Volumes/Expansion/datasets/porto_taxi
mkdir -p /Volumes/Expansion/datasets/nyc_yellow_taxi_2013
mkdir -p /Volumes/Expansion/datasets/chicago_taxi

echo "Downloading Porto Taxi (UCI)..."
curl -L -C - "https://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip" -o "/Volumes/Expansion/datasets/porto_taxi/porto_taxi.zip"
unzip -n "/Volumes/Expansion/datasets/porto_taxi/porto_taxi.zip" -d "/Volumes/Expansion/datasets/porto_taxi/"

echo "Downloading NYC Yellow Taxi Jan 2013 (Classic DDC benchmark with Hack IDs)..."
curl -L -C - "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2013-01.parquet" -o "/Volumes/Expansion/datasets/nyc_yellow_taxi_2013/yellow_tripdata_2013-01.parquet"

echo "Downloading Chicago Taxi (100k sample)..."
curl -L -C - "https://data.cityofchicago.org/resource/wrvz-psew.csv?\$limit=100000" -o "/Volumes/Expansion/datasets/chicago_taxi/chicago_taxi_sample.csv"

echo "Cloning Shanghai Taxi AIRL Repo for their dataset..."
cd /Volumes/Expansion/datasets/
if [ ! -d "shanghai_taxi_rcm_airl" ]; then
    git clone https://github.com/liangchunyaobing/RCM-AIRL.git shanghai_taxi_rcm_airl
fi

echo "--- DIRECTORY SIZES ---"
ls -lh /Volumes/Expansion/datasets/porto_taxi
ls -lh /Volumes/Expansion/datasets/nyc_yellow_taxi_2013
ls -lh /Volumes/Expansion/datasets/chicago_taxi
ls -lh /Volumes/Expansion/datasets/shanghai_taxi_rcm_airl/data
