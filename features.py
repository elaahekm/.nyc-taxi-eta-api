import numpy as np
import pandas as pd

df = pd.read_parquet("data/clean.parquet")
centers = pd.read_csv("data/zone_centers.csv").set_index("LocationID")

df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

df = df.sample(1_000_000, random_state=42)

# distance between pickup and dropoff zone centers (haversine, km)
lat1 = np.radians(df["PULocationID"].map(centers["lat"]))
lon1 = np.radians(df["PULocationID"].map(centers["lon"]))
lat2 = np.radians(df["DOLocationID"].map(centers["lat"]))
lon2 = np.radians(df["DOLocationID"].map(centers["lon"]))

a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
df["distance_km"] = 6371 * 2 * np.arcsin(np.sqrt(a))

print("rows with missing distance:", df["distance_km"].isna().sum())
df = df.dropna(subset=["distance_km"])

df = df[["PULocationID", "DOLocationID", "hour", "day_of_week", "is_weekend", "distance_km", "duration_min"]]

print(df.shape)
print(df["distance_km"].describe())

df.to_parquet("data/features.parquet")
print("saved data/features.parquet")
