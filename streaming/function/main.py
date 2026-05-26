"""
Google Cloud Function — Student Habits streaming prediction worker.

Triggered by a Pub/Sub message containing a single student's habit data,
this function loads the trained model (from MLflow) and the preprocessing
artefacts (from GCS), runs a prediction, and publishes the result to a
second Pub/Sub topic that the UI listens to.

Mirrors the structure of the teacher's Iris example
(04_streaming/function/main.py) but adapted for:
  - 14 features instead of 4
  - integer-encoded categorical fields (gender, diet_quality, …)
  - a continuous regression output (exam_score 0–100) instead of a class id
"""

import base64
import json
import os
import tempfile

import joblib
import mlflow.pyfunc
from google.cloud import pubsub_v1
from google.cloud import storage


# ──────────────────────────────────────────
# Environment variables (set when deploying the function)
# ──────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT")
OUTPUT_TOPIC = os.environ.get("PREDICTIONS_TOPIC", "student-habits-predictions")

# MLflow model URI, e.g. "models:/student-habits-model/Production"
# or "runs:/<run_id>/model".
MODEL_URI = os.environ.get("MODEL_URI")

# GCS URIs for the preprocessing artefacts saved by train.py / schedule.py.
CATEGORY_MAPS_URI = os.environ.get("CATEGORY_MAPS_URI")
FEATURE_COLUMNS_URI = os.environ.get("FEATURE_COLUMNS_URI")


# ──────────────────────────────────────────
# Globals — populated lazily on first invocation, then reused across calls
# in the same container (Cloud Functions keeps warm instances around).
# ──────────────────────────────────────────
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC) if PROJECT_ID else None

model = None
category_maps = None
feature_columns = None


# Sensible clip range for the regression target.
SCORE_MIN = 0.0
SCORE_MAX = 100.0


def download_blob_from_gcs(gcs_uri: str, local_path: str):
    """Download a single object from GCS to a local file path."""
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    uri_without_prefix = gcs_uri[len("gs://"):]
    bucket_name, blob_name = uri_without_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)


def get_model():
    """Lazy-load the MLflow model on first invocation, cache on subsequent ones."""
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    return model


def get_category_maps():
    """Lazy-load category_maps.json from GCS."""
    global category_maps
    if category_maps is None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            local_path = tmp_file.name
        download_blob_from_gcs(CATEGORY_MAPS_URI, local_path)
        with open(local_path, "r", encoding="utf-8") as f:
            category_maps = json.load(f)
    return category_maps


def get_feature_columns():
    """Lazy-load feature_columns.joblib from GCS."""
    global feature_columns
    if feature_columns is None:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            local_path = tmp_file.name
        download_blob_from_gcs(FEATURE_COLUMNS_URI, local_path)
        feature_columns = joblib.load(local_path)
    return feature_columns


def encode_record(record: dict, maps: dict):
    """
    Apply the same categorical encoding as train.py:
      - strip + title-case raw text values
      - map them to the integers stored in category_maps.json

    Returns a new dict with all categorical fields replaced by ints.
    """
    encoded = dict(record)
    for column, mapping in maps.items():
        if column not in encoded:
            continue
        raw_value = encoded[column]
        if isinstance(raw_value, str):
            raw_value = raw_value.strip().title()
        if raw_value not in mapping:
            raise ValueError(
                f"Invalid value '{encoded[column]}' for field '{column}'. "
                f"Allowed: {list(mapping.keys())}"
            )
        encoded[column] = mapping[raw_value]
    return encoded


def prepare_features(record: dict):
    """
    Encode the categorical fields and return a 2D list in the exact column
    order the model was trained with.
    """
    encoded = encode_record(record, get_category_maps())
    columns = get_feature_columns()

    missing = [c for c in columns if c not in encoded]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return [[encoded[col] for col in columns]]


def clip_score(score: float) -> float:
    """Keep the prediction inside the valid exam-score range."""
    return max(SCORE_MIN, min(SCORE_MAX, float(score)))


def predict_student_habits(cloud_event):
    """
    Cloud Function entry point.

    Expects a Pub/Sub-formatted CloudEvent whose decoded payload looks like:
        {
            "student_id": "<uuid>",
            "features": {
                "age": 20,
                "gender": "Male",
                "study_hours_per_day": 6.0,
                ...
            }
        }
    """
    message_data = cloud_event.data["message"]["data"]
    decoded = base64.b64decode(message_data).decode("utf-8")
    event = json.loads(decoded)

    print("Function triggered")
    print(event)

    student_id = event.get("student_id")
    features = event.get("features", {})

    try:
        X = prepare_features(features)
        raw_score = float(get_model().predict(X)[0])
        score = round(clip_score(raw_score), 2)

        response = {
            "prediction": {
                "student_id": student_id,
                "predicted_exam_score": score,
            }
        }
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        response = {
            "prediction": {
                "student_id": student_id,
                "error": str(exc),
            }
        }

    publisher.publish(
        topic_path,
        json.dumps(response).encode("utf-8"),
    )
    print("Published response:", response)
