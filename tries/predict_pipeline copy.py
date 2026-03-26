# predict_pipeline.py
# FastAPI inference service for the Student Habits → Exam Score predictor.
#
# Start the server (after running train_pipeline.py at least once):
#   export MLFLOW_TRACKING_URI=http://localhost:5000   # or file:./mlruns
#   uvicorn predict_pipeline:app --reload --port 8000
#
# Then open  http://127.0.0.1:8000/docs  for the interactive Swagger UI.

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── MLflow setup ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Load artefacts produced by train_pipeline.py ────────────────────────────
try:
    scaler = joblib.load("scaler.joblib")
    model  = joblib.load("best_model.joblib")
    print("[startup] Loaded scaler.joblib and best_model.joblib")
except FileNotFoundError as exc:
    raise RuntimeError(
        "Could not find scaler.joblib / best_model.joblib. "
        "Please run train_pipeline.py first."
    ) from exc


# ── Feature schema ────────────────────────────────────────────────────────────
# All fields match the columns used during training (after encoding).
# Categorical fields expect the integer encoding used in train_pipeline.py:
#   gender:                      0=Male, 1=Female, 2=Other
#   diet_quality:                0=Poor, 1=Fair, 2=Good
#   parental_education_level:    0=High School, 1=Bachelor, 2=Master
#   internet_quality:            0=Poor, 1=Average, 2=Good
#   extracurricular_participation: 0=No, 1=Yes
#   part_time_job:               0=No, 1=Yes

class StudentFeatures(BaseModel):
    age: float                          = Field(..., ge=10, le=100,  example=20)
    gender: int                         = Field(..., ge=0,  le=2,    example=0,  description="0=Male 1=Female 2=Other")
    study_hours_per_day: float          = Field(..., ge=0,  le=24,   example=5.0)
    social_media_hours: float           = Field(..., ge=0,  le=24,   example=2.0)
    netflix_hours: float                = Field(..., ge=0,  le=24,   example=1.5)
    part_time_job: int                  = Field(..., ge=0,  le=1,    example=0,  description="0=No 1=Yes")
    attendance_percentage: float        = Field(..., ge=0,  le=100,  example=85.0)
    sleep_hours: float                  = Field(..., ge=0,  le=24,   example=7.0)
    diet_quality: int                   = Field(..., ge=0,  le=2,    example=1,  description="0=Poor 1=Fair 2=Good")
    exercise_frequency: float           = Field(..., ge=0,  le=7,    example=3.0)
    parental_education_level: int       = Field(..., ge=0,  le=2,    example=1,  description="0=High School 1=Bachelor 2=Master")
    internet_quality: int               = Field(..., ge=0,  le=2,    example=2,  description="0=Poor 1=Average 2=Good")
    mental_health_rating: float         = Field(..., ge=1,  le=10,   example=7.0)
    extracurricular_participation: int  = Field(..., ge=0,  le=1,    example=1,  description="0=No 1=Yes")


class PredictionResponse(BaseModel):
    predicted_exam_score: float
    model_type: str


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Habits → Exam Score Predictor",
    description=(
        "Predicts a student's exam score from their daily habits and background. "
        "Run train_pipeline.py first to generate the required model artefacts."
    ),
    version="1.0.0",
)


# ── Helpers ───────────────────────────────────────────────────────────────────
FEATURE_ORDER = [
    "age",
    "gender",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "part_time_job",
    "attendance_percentage",
    "sleep_hours",
    "diet_quality",
    "exercise_frequency",
    "parental_education_level",
    "internet_quality",
    "mental_health_rating",
    "extracurricular_participation",
]


def features_to_array(features: StudentFeatures) -> np.ndarray:
    """Convert the Pydantic model to a scaled numpy row ready for inference."""
    row = pd.DataFrame([features.dict()])[FEATURE_ORDER]
    row_scaled = scaler.transform(row)
    return row_scaled


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "message": "Student Exam Score Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse, summary="Predict exam score")
def predict(features: StudentFeatures):
    """
    Submit a student's features and receive a predicted exam score.

    - All categorical inputs must be supplied as integers (see field descriptions).
    - The scaler and model are loaded once at startup from the artefacts produced
      by `train_pipeline.py`.
    """
    try:
        X = features_to_array(features)
        score = float(model.predict(X)[0])
        # Clamp to a sensible range
        score = max(0.0, min(100.0, score))
        model_name = type(model).__name__
        return PredictionResponse(
            predicted_exam_score=round(score, 2),
            model_type=model_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model-info", summary="Show loaded model type")
def model_info():
    """Return information about the currently loaded model."""
    return {
        "model_type": type(model).__name__,
        "model_params": model.get_params(),
        "scaler_type": type(scaler).__name__,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
    }
