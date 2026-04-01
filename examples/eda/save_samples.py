from sample_datasets import sample_tdrive, sample_citibike, sample_foursquare

tdrive = sample_tdrive(n_files=10, n_rows_per_file=500)
if tdrive is not None:
    tdrive.to_csv("data/samples/tdrive_sample.csv", index=False)
    print("Saved T-Drive sample to data/samples/tdrive_sample.csv")

citibike = sample_citibike(n_rows=5000)
if citibike is not None:
    citibike.to_csv("data/samples/citibike_sample.csv", index=False)
    print("Saved CitiBike sample to data/samples/citibike_sample.csv")

foursquare = sample_foursquare(n_rows=5000)
if foursquare is not None:
    foursquare.to_csv("data/samples/foursquare_sample.csv", index=False)
    print("Saved Foursquare sample to data/samples/foursquare_sample.csv")
