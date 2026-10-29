import pandas as pd

df = pd.read_parquet(
    "data/yellow_tripdata_2015-01.parquet",
    columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"],
)
before = len(df)

df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

# keep only January 2015
df = df[(df["tpep_pickup_datetime"] >= "2015-01-01") & (df["tpep_pickup_datetime"] < "2015-02-01")]
# keep only reasonable trips: 1 to 180 minutes
df = df[(df["duration_min"] >= 1) & (df["duration_min"] <= 180)]

print("rows before:", before)
print("rows after:", len(df))
print("removed:", before - len(df))
print(df["duration_min"].describe())

df.to_parquet("data/clean.parquet")
print("saved data/clean.parquet")
