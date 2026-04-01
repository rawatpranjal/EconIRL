import pandas as pd
df = pd.read_csv("data/raw/foursquare/dataset_TSMC2014_NYC.csv", nrows=1000)
print(df.columns)
print(df.nunique())
print(df[['userId', 'venueCategory', 'utcTimestamp']].head(10))
