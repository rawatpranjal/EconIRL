import os
import urllib.request
import urllib.error
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

base_dir = "/Volumes/Expansion/datasets/"
os.makedirs(os.path.join(base_dir, "nyc_tlc"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "beijing_osm"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "weather"), exist_ok=True)

tlc_urls = {
    "yellow_tripdata_2024-01.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    "fhvhv_tripdata_2024-01.parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet"
}

print("Downloading NYC TLC Datasets (Gig-Economy Earnings)...")
for filename, url in tlc_urls.items():
    out_path = os.path.join(base_dir, "nyc_tlc", filename)
    if not os.path.exists(out_path):
        print(f" Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f" Successfully downloaded {filename} ({os.path.getsize(out_path)/1e6:.1f} MB)")
        except Exception as e:
            print(f" Failed to download {filename}: {e}")
    else:
        print(f" {filename} already exists. ({os.path.getsize(out_path)/1e6:.1f} MB) Skipping.")
