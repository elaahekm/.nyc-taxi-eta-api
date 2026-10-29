from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Snapp ETA API")

model = joblib.load("eta_model_v2.joblib")


class TripRequest(BaseModel):
    pickup_zone: int = Field(..., ge=1, le=265, description="Pickup zone ID (1-265)")
    dropoff_zone: int = Field(..., ge=1, le=265, description="Dropoff zone ID (1-265)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")


@app.post("/predict_eta")
def predict_eta(request: TripRequest):
    input_data = pd.DataFrame([{
        "PULocationID": request.pickup_zone,
        "DOLocationID": request.dropoff_zone,
        "hour": request.hour,
        "day_of_week": request.day_of_week,
        "is_weekend": int(request.day_of_week >= 5),
    }])

    prediction = model.predict(input_data)[0]

    return {"estimated_time_minutes": round(float(prediction), 2)}