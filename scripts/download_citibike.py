"""Download and preprocess Citibike trip data for econirl.

Downloads one month of Citibike trip data, clusters stations into
groups by geographic proximity, discretizes time of day, and produces
two processed datasets:
  1. Route choice: station-to-station OD trips discretized by cluster
  2. Usage frequency: member-day panel of ride/no-ride decisions

Usage:
    python scripts/download_citibike.py [--month 2024-01] [--output-dir data/processed]

The raw data comes from the Citibike System Data page. Files are
hosted on S3 and available under a non-commercial research license.
"""

import argparse
import os
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

def download_raw(month: str, output_dir: Path) -> Path:
    """Download raw Citibike CSV zip for a given month.

    Tries multiple URL formats since Citibike changed naming conventions:
    - 2024+: 202401-citibike-tripdata.zip (no dash, no .csv)
    - 2023: 202301-citibike-tripdata.csv.zip (no dash, with .csv)
    - older: 2023-01-citibike-tripdata.csv.zip (with dash)
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Normalize month: "2024-01" -> "202401"
    month_compact = month.replace("-", "")

    zip_path = raw_dir / f"{month_compact}-citibike-tripdata.zip"
    if zip_path.exists():
        print(f"  Raw file already exists: {zip_path}")
        return zip_path

    url_candidates = [
        f"https://s3.amazonaws.com/tripdata/{month_compact}-citibike-tripdata.zip",
        f"https://s3.amazonaws.com/tripdata/{month_compact}-citibike-tripdata.csv.zip",
        f"https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.csv.zip",
        f"https://s3.amazonaws.com/tripdata/JC-{month}-citibike-tripdata.csv.zip",
    ]

    for url in url_candidates:
        print(f"  Trying {url}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            size_mb = os.path.getsize(zip_path) / 1e6
            print(f"  Downloaded {size_mb:.1f} MB")
            return zip_path
        except Exception:
            continue

    raise RuntimeError(
        f"Could not download Citibike data for {month}. "
        f"Check https://s3.amazonaws.com/tripdata/index.html for available files."
    )


def load_and_clean(zip_path: Path) -> pd.DataFrame:
    """Load and clean raw Citibike trip data."""
    import zipfile

    print("  Loading CSV from ZIP...")
    with zipfile.ZipFile(zip_path) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if len(csv_files) == 1:
            df = pd.read_csv(zf.open(csv_files[0]), low_memory=False)
        else:
            print(f"  Found {len(csv_files)} CSV files in ZIP, concatenating...")
            df = pd.concat(
                [pd.read_csv(zf.open(f), low_memory=False) for f in sorted(csv_files)],
                ignore_index=True,
            )

    # Standardize column names (format changed over time)
    col_map = {}
    for col in df.columns:
        lower = col.lower().replace(" ", "_")
        col_map[col] = lower
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    required = ["start_lat", "start_lng", "end_lat", "end_lng", "started_at", "member_casual"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    # Drop rows with missing coordinates
    df = df.dropna(subset=["start_lat", "start_lng", "end_lat", "end_lng"])

    # Filter to valid coordinate ranges (NYC area)
    df = df[
        (df["start_lat"].between(40.5, 41.0))
        & (df["start_lng"].between(-74.3, -73.7))
        & (df["end_lat"].between(40.5, 41.0))
        & (df["end_lng"].between(-74.3, -73.7))
    ]

    # Parse timestamps
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])

    print(f"  Loaded {len(df):,} trips after cleaning")
    return df


def cluster_stations(df: pd.DataFrame, n_clusters: int = 20) -> tuple[pd.DataFrame, np.ndarray]:
    """Cluster stations by geographic proximity using K-Means."""
    # Collect all unique station locations
    start_coords = df[["start_lat", "start_lng"]].drop_duplicates().values
    end_coords = df[["end_lat", "end_lng"]].drop_duplicates().values
    all_coords = np.vstack([start_coords, end_coords])
    all_coords = np.unique(all_coords, axis=0)

    print(f"  Clustering {len(all_coords)} unique locations into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(all_coords)

    # Assign clusters to trips
    df = df.copy()
    df["origin_cluster"] = kmeans.predict(df[["start_lat", "start_lng"]].values)
    df["dest_cluster"] = kmeans.predict(df[["end_lat", "end_lng"]].values)

    return df, kmeans.cluster_centers_


def build_route_choice(df: pd.DataFrame, centroids: np.ndarray, n_clusters: int = 20) -> pd.DataFrame:
    """Build route choice dataset: origin cluster + time -> destination cluster."""
    df = df.copy()

    # Time of day buckets
    hour = df["started_at"].dt.hour
    df["time_bucket"] = pd.cut(
        hour,
        bins=[0, 6, 12, 18, 24],
        labels=[0, 1, 2, 3],
        right=False,
        include_lowest=True,
    ).astype(int)

    # State = origin_cluster * 4 + time_bucket
    df["state"] = df["origin_cluster"] * 4 + df["time_bucket"]
    df["action"] = df["dest_cluster"]

    # Compute features for each origin-destination pair
    records = []
    for _, row in df.iterrows():
        oc = int(row["origin_cluster"])
        dc = int(row["dest_cluster"])
        tb = int(row["time_bucket"])

        # Euclidean distance between centroids (approximate, lat/lng)
        dist = np.sqrt(
            (centroids[oc, 0] - centroids[dc, 0]) ** 2
            + (centroids[oc, 1] - centroids[dc, 1]) ** 2
        )

        records.append({
            "trip_idx": len(records),
            "state": int(row["state"]),
            "action": dc,
            "origin_cluster": oc,
            "dest_cluster": dc,
            "time_bucket": tb,
            "distance": dist,
            "is_peak": 1 if tb in [1, 2] else 0,
            "started_at": row["started_at"],
        })

    route_df = pd.DataFrame(records)

    # Compute destination popularity
    dest_counts = route_df["dest_cluster"].value_counts(normalize=True)
    route_df["dest_popularity"] = route_df["dest_cluster"].map(dest_counts)

    # Subsample to manageable size
    if len(route_df) > 50000:
        route_df = route_df.sample(n=50000, random_state=42).reset_index(drop=True)
        route_df["trip_idx"] = range(len(route_df))

    return route_df


def build_usage_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Build usage frequency panel: member-day ride/no-ride decisions."""
    # Filter to members only (casual riders have no persistent ID)
    members = df[df["member_casual"] == "member"].copy()

    # Create a rider ID from start station (proxy since no real rider ID)
    # Group by date and approximate rider identity via station patterns
    members["date"] = members["started_at"].dt.date
    members["day_of_week"] = members["started_at"].dt.dayofweek

    # Aggregate daily rides per approximate rider
    # Use start_station_id as a proxy for rider identity
    if "start_station_id" in members.columns:
        rider_col = "start_station_id"
    else:
        rider_col = "origin_cluster"

    daily = members.groupby([rider_col, "date"]).agg(
        n_rides=("started_at", "count"),
        day_of_week=("day_of_week", "first"),
    ).reset_index()
    daily.rename(columns={rider_col: "rider_id"}, inplace=True)

    # Build full panel (all rider-day combinations)
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max())
    riders = daily["rider_id"].unique()

    # Subsample riders for tractability
    if len(riders) > 500:
        rng = np.random.RandomState(42)
        riders = rng.choice(riders, size=500, replace=False)
        daily = daily[daily["rider_id"].isin(riders)]

    panel_rows = []
    for rid in riders:
        rider_data = daily[daily["rider_id"] == rid].set_index("date")
        rides_last_7 = 0
        for dt in all_dates:
            day_type = 0 if dt.dayofweek < 5 else 1  # weekday=0, weekend=1
            rode = 1 if dt.date() in rider_data.index else 0

            # Recent usage bucket: rides in last 7 days
            if rides_last_7 == 0:
                usage_bucket = 0
            elif rides_last_7 <= 2:
                usage_bucket = 1
            elif rides_last_7 <= 5:
                usage_bucket = 2
            else:
                usage_bucket = 3

            state = day_type * 4 + usage_bucket
            panel_rows.append({
                "rider_id": rid,
                "date": dt.date(),
                "state": state,
                "action": rode,
                "day_type": day_type,
                "usage_bucket": usage_bucket,
                "rides_last_7": rides_last_7,
            })

            # Update rolling count
            if rode:
                rides_last_7 = min(rides_last_7 + 1, 7)
            else:
                rides_last_7 = max(rides_last_7 - 1, 0)

    usage_df = pd.DataFrame(panel_rows)

    # Add next_state
    usage_df["next_state"] = usage_df.groupby("rider_id")["state"].shift(-1)
    usage_df = usage_df.dropna(subset=["next_state"])
    usage_df["next_state"] = usage_df["next_state"].astype(int)

    return usage_df


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess Citibike data")
    parser.add_argument("--month", default="2024-01", help="Month to download (YYYY-MM)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--n-clusters", type=int, default=20, help="Number of station clusters")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Citibike Data Pipeline: {args.month}")
    print("=" * 50)

    # Download
    print("\n1. Downloading raw data...")
    zip_path = download_raw(args.month, output_dir)

    # Load and clean
    print("\n2. Loading and cleaning...")
    df = load_and_clean(zip_path)

    # Cluster stations
    print("\n3. Clustering stations...")
    df, centroids = cluster_stations(df, n_clusters=args.n_clusters)

    # Build route choice dataset
    print("\n4. Building route choice dataset...")
    route_df = build_route_choice(df, centroids, n_clusters=args.n_clusters)
    route_path = output_dir / "citibike_route.csv"
    route_df.to_csv(route_path, index=False)
    print(f"  Saved {len(route_df):,} trips to {route_path}")

    # Save centroids
    centroids_path = output_dir / "citibike_centroids.npy"
    np.save(centroids_path, centroids)
    print(f"  Saved {len(centroids)} centroids to {centroids_path}")

    # Build usage panel
    print("\n5. Building usage frequency panel...")
    usage_df = build_usage_panel(df)
    usage_path = output_dir / "citibike_usage.csv"
    usage_df.to_csv(usage_path, index=False)
    print(f"  Saved {len(usage_df):,} rider-day observations to {usage_path}")

    print("\nDone. Files saved to:")
    print(f"  Route choice:    {route_path}")
    print(f"  Station clusters: {centroids_path}")
    print(f"  Usage panel:     {usage_path}")


if __name__ == "__main__":
    main()
