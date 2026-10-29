import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("data/features.parquet")

features = ["PULocationID", "DOLocationID", "hour", "day_of_week", "is_weekend"]
X = df[features]
y = df["duration_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(loss="absolute_error", random_state=42)
print("training...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("baseline MAE (min): 6.24")
print("model MAE (min):", round(mae, 2))

joblib.dump(model, "eta_model_v2.joblib")
print("saved eta_model_v2.joblib")