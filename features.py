import pandas as pd

df = pd.read_parquet("data/clean.parquet")

df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["day_of_week"] = df["tpep_pickup_datetime"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df = df.sample(1_000_000, random_state=42)

df = df[["PULocationID", "DOLocationID", "hour", "day_of_week", "is_weekend", "duration_min"]]

print(df.shape)
print(df.head())
print(df.groupby("hour")["duration_min"].mean().round(1))

df.to_parquet("data/features.parquet")
print("saved data/features.parquet")
