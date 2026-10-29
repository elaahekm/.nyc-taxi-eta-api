import pandas as pd

df = pd.read_parquet(
    "data/yellow_tripdata_2015-01.parquet",
    columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"],
)
df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

print(df["duration_min"].describe())
print("سفر با مدت صفر یا منفی:", (df["duration_min"] <= 0).sum())
print("سفر بیشتر از ۳ ساعت:", (df["duration_min"] > 180).sum())
print("اولین و آخرین تاریخ:", df["tpep_pickup_datetime"].min(), df["tpep_pickup_datetime"].max())
