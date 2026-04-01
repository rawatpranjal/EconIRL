import urllib.request
import os
import zipfile
import ssl

out_dir = "/Volumes/Expansion/datasets/nyc_yellow_taxi_2013_original"
os.makedirs(out_dir, exist_ok=True)

urls = {
    "trip_data_1.zip": "https://archive.org/download/nycTaxiTripData2013/trip_data_1.zip",
    "trip_fare_1.zip": "https://archive.org/download/nycTaxiTripData2013/trip_fare_1.zip"
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

for filename, url in urls.items():
    out_path = os.path.join(out_dir, filename)
    if not os.path.exists(out_path):
        print(f"Downloading {filename} from Archive.org...")
        with urllib.request.urlopen(url, context=ctx) as response, open(out_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Saved to {out_path}")
    else:
        print(f"{filename} already exists.")

print("Extracting headers to verify hack_license...")
for filename in urls.keys():
    z_path = os.path.join(out_dir, filename)
    try:
        with zipfile.ZipFile(z_path, 'r') as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                header = f.readline().decode('utf-8').strip()
                print(f"{csv_name} header: {header}")
    except Exception as e:
        print(f"Could not read zip {filename}: {e}")

