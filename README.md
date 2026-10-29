# NYC Taxi Trip Duration (ETA) Prediction API

An end-to-end machine learning project: it cleans real taxi trip data, trains a model that predicts trip duration, and serves predictions through a FastAPI service packaged with Docker.

## Results

Trained on a random sample of 1,000,000 yellow taxi trips (January 2015), evaluated on a held-out 20% test set (200,000 trips).

| Model | MAE (minutes) |
|---|---|
| Baseline (always predict the median, ~10 min) | 6.24 |
| HistGradientBoostingRegressor | 4.60 |

The model reduces the average error by about 26% compared to the baseline.

## Data

- Source: NYC TLC Trip Record Data, Yellow Taxi, January 2015 (`yellow_tripdata_2015-01.parquet`).
- 12,741,035 raw trips. After cleaning (only January 2015 pickups, durations between 1 and 180 minutes), 12,630,986 trips remain.
- This version of the data has pickup and dropoff **zone IDs**, not GPS coordinates, so the model uses zones.
- The data file is not included in this repository. Download it from the NYC TLC website and place it in `data/`.

## Features

`PULocationID`, `DOLocationID`, `hour`, `day_of_week`, `is_weekend`

## Project structure

```
clean.py          # remove invalid trips, save data/clean.parquet
features.py       # build features, sample 1M rows, save data/features.parquet
baseline.py       # median baseline and its MAE
train_model.py    # train the model and save eta_model_v2.joblib
main.py           # FastAPI service
Dockerfile
requirements.txt
```

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt pyarrow

python3 clean.py
python3 features.py
python3 baseline.py
python3 train_model.py

uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs to try the API.

### Example request

```bash
curl -X POST http://127.0.0.1:8000/predict_eta \
  -H "Content-Type: application/json" \
  -d '{"pickup_zone": 132, "dropoff_zone": 237, "hour": 18, "day_of_week": 2}'
```

Response:

```json
{"estimated_time_minutes": 38.51}
```

`day_of_week`: 0 = Monday, 6 = Sunday.

## Docker

```bash
docker build -t eta-api .
docker run -p 8000:8000 eta-api
```

Library versions are pinned in `requirements.txt` so the saved model loads the same way inside the container.

## Limitations

- Only one month of data, so the model does not capture seasonal effects.
- The model knows the pickup and dropoff zones but not the distance between them.
- No weather or live traffic information.
- No automated tests yet, and the service is not deployed publicly.

## Ideas for improvement

- Add the distance between zone centers as a feature.
- Train on several months and compare more models.
- Analyze where the model makes its largest errors (for example airport trips).
- Add automated tests and deploy the API.