import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Snapp ETA API")

model = joblib.load("eta_model_v3.joblib")
centers = pd.read_csv("assets/zone_centers.csv").set_index("LocationID")


class TripRequest(BaseModel):
    pickup_zone: int = Field(..., ge=1, le=265, description="Pickup zone ID (1-265)")
    dropoff_zone: int = Field(..., ge=1, le=265, description="Dropoff zone ID (1-265)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@app.post("/predict_eta")
def predict_eta(request: TripRequest):
    if request.pickup_zone not in centers.index or request.dropoff_zone not in centers.index:
        raise HTTPException(status_code=422, detail="Unknown zone ID")

    pu = centers.loc[request.pickup_zone]
    do = centers.loc[request.dropoff_zone]
    distance_km = float(haversine_km(pu["lat"], pu["lon"], do["lat"], do["lon"]))

    input_data = pd.DataFrame([{
        "PULocationID": request.pickup_zone,
        "DOLocationID": request.dropoff_zone,
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "is_weekend": int(request.day_of_week >= 5),
        "distance_km": distance_km,
    }])

    prediction = model.predict(input_data)[0]

    return {
        "distance_km": round(distance_km, 2),
        "estimated_time_minutes": round(float(prediction), 2),
    }
