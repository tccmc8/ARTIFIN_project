# Streaming Prediction Function

A Google Cloud Function that performs **event-driven streaming inference**
on incoming student-habit data. This is the streaming counterpart to the
synchronous HTTP API in `app/`.

## Architecture

```
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│  Pub/Sub topic      │      │  Cloud Function      │      │  Pub/Sub topic       │
│  student-habits-    │ ───▶ │  predict_student_    │ ───▶ │  student-habits-     │
│  requests           │      │  habits              │      │  predictions         │
└─────────────────────┘      └──────────────────────┘      └──────────────────────┘
                                       │
                                       ▼
                             Loads model from MLflow
                             Loads encoders from GCS
```

The function is triggered every time a message is published to the
`student-habits-requests` topic. It loads the trained model from the
MLflow registry, pulls the preprocessing artefacts (`category_maps`,
`feature_columns`) from Google Cloud Storage, runs a prediction, and
publishes the result back to the `student-habits-predictions` topic.

## Relationship to the UI

The original streaming architecture had the UI (`streaming/ui/`)
subscribe to the predictions topic. For the live demo this was simplified
to direct HTTP — the UI now POSTs to the synchronous API in `app/` and
displays the response immediately. The function in this folder remains as
the reference implementation of the streaming pattern.

## Entry Point

`predict_student_habits(cloud_event)` — the Cloud Event payload contains
the Pub/Sub message with a single student's habits as JSON.

## Required Environment Variables

| Variable | Purpose | Example |
|---|---|---|
| `GCP_PROJECT` | Project ID for publishing results | `YOUR-PROJECT-ID` |
| `PREDICTIONS_TOPIC` | Output Pub/Sub topic | `student-habits-predictions` |
| `MODEL_URI` | MLflow model URI | `models:/student-habits-model/Production` |
| `CATEGORY_MAPS_URI` | GCS path to encoding map | `gs://my-bucket/category_maps.json` |
| `FEATURE_COLUMNS_URI` | GCS path to feature list | `gs://my-bucket/feature_columns.joblib` |

## Deployment

Before deploying, ensure the two Pub/Sub topics exist and the
preprocessing artefacts have been uploaded to GCS:

```bash
# One-time setup
gcloud pubsub topics create student-habits-requests
gcloud pubsub topics create student-habits-predictions
gsutil cp models/category_maps.json   gs://YOUR-BUCKET/category_maps.json
gsutil cp models/feature_columns.joblib gs://YOUR-BUCKET/feature_columns.joblib
```

Then deploy the function (2nd generation Cloud Functions, Pub/Sub trigger):

```bash
gcloud functions deploy student-habits-streaming \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=streaming/function \
  --entry-point=predict_student_habits \
  --trigger-topic=student-habits-requests \
  --set-env-vars="GCP_PROJECT=YOUR-PROJECT-ID,PREDICTIONS_TOPIC=student-habits-predictions,MODEL_URI=models:/student-habits-model/Production,CATEGORY_MAPS_URI=gs://YOUR-BUCKET/category_maps.json,FEATURE_COLUMNS_URI=gs://YOUR-BUCKET/feature_columns.joblib"
```

Run from the project root. Cloud Functions builds the container itself
from the contents of `streaming/function/`, using `requirements.txt` for
Python dependencies. No Dockerfile is required.

## Testing the Function

Publish a sample message to the input topic:

```bash
gcloud pubsub topics publish student-habits-requests --message='{
  "age": 20,
  "gender": "Female",
  "study_hours_per_day": 6.5,
  "social_media_hours": 2.0,
  "netflix_hours": 1.0,
  "part_time_job": "No",
  "attendance_percentage": 92,
  "sleep_hours": 7.5,
  "diet_quality": "Good",
  "exercise_frequency": 3,
  "parental_education_level": "Bachelor",
  "internet_quality": "Good",
  "mental_health_rating": 8,
  "extracurricular_participation": "Yes"
}'
```

Then pull the prediction from the output topic:

```bash
gcloud pubsub subscriptions create predictions-test \
  --topic=student-habits-predictions
gcloud pubsub subscriptions pull predictions-test --auto-ack --limit=5
```

## Removing the Function

```bash
gcloud functions delete student-habits-streaming --region=us-central1 --gen2
gcloud pubsub topics       delete student-habits-requests
gcloud pubsub topics       delete student-habits-predictions
gcloud pubsub subscriptions delete predictions-test
```
