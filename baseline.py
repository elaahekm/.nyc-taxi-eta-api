import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("data/features.parquet")

train, test = train_test_split(df, test_size=0.2, random_state=42)

median_value = train["duration_min"].median()
predictions = np.full(len(test), median_value)

mae = mean_absolute_error(test["duration_min"], predictions)

print("train rows:", len(train))
print("test rows:", len(test))
print("baseline median (min):", round(median_value, 2))
print("baseline MAE (min):", round(mae, 2))