# predict_pipeline.py
# FastAPI inference service for the Student Habits -> Exam Score predictor.
#
# Start (after running train_pipeline.py or the notebook first):
#   uvicorn predict_pipeline:app --reload --port 8000
#
# Swagger UI: http://127.0.0.1:8000/docs

import pathlib
import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Absolute paths ────────────────────────────────────────────────────────────
PROJECT_DIR   = pathlib.Path(__file__).resolve().parent
DB_PATH       = PROJECT_DIR / "mlflow.db"
ARTIFACT_DIR  = PROJECT_DIR / "mlartifacts"
SCALER_PATH   = PROJECT_DIR / "scaler.joblib"
MODEL_PATH    = PROJECT_DIR / "best_model.joblib"

MLFLOW_TRACKING_URI  = f"sqlite:///{DB_PATH}"
MLFLOW_ARTIFACT_ROOT = ARTIFACT_DIR.as_uri()

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ── Load scaler ───────────────────────────────────────────────────────────────
if not SCALER_PATH.exists():
    raise RuntimeError(f"scaler.joblib not found at {SCALER_PATH}. Run train_pipeline.py first.")
scaler = joblib.load(SCALER_PATH)
print(f"[startup] Loaded scaler from {SCALER_PATH}")

# ── Load model ────────────────────────────────────────────────────────────────
try:
    model = mlflow.sklearn.load_model("models:/BestStudentHabitsModel@Staging")
    model_source = "MLflow registry (BestStudentHabitsModel@Staging)"
except Exception:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"best_model.joblib not found at {MODEL_PATH}. Run train_pipeline.py first.")
    model = joblib.load(MODEL_PATH)
    model_source = f"joblib ({MODEL_PATH.name})"
print(f"[startup] Loaded model from {model_source}")


# ── Feature schema ────────────────────────────────────────────────────────────
class StudentFeatures(BaseModel):
    age: float                         = Field(..., ge=10,  le=100, example=20)
    gender: int                        = Field(..., ge=0,   le=2,   example=0,   description="0=Male 1=Female 2=Other")
    study_hours_per_day: float         = Field(..., ge=0,   le=24,  example=5.0)
    social_media_hours: float          = Field(..., ge=0,   le=24,  example=2.0)
    netflix_hours: float               = Field(..., ge=0,   le=24,  example=1.5)
    part_time_job: int                 = Field(..., ge=0,   le=1,   example=0,   description="0=No 1=Yes")
    attendance_percentage: float       = Field(..., ge=0,   le=100, example=85.0)
    sleep_hours: float                 = Field(..., ge=0,   le=24,  example=7.0)
    diet_quality: int                  = Field(..., ge=0,   le=2,   example=1,   description="0=Poor 1=Fair 2=Good")
    exercise_frequency: float          = Field(..., ge=0,   le=7,   example=3.0)
    parental_education_level: int      = Field(..., ge=0,   le=2,   example=1,   description="0=High School 1=Bachelor 2=Master")
    internet_quality: int              = Field(..., ge=0,   le=2,   example=2,   description="0=Poor 1=Average 2=Good")
    mental_health_rating: float        = Field(..., ge=1,   le=10,  example=7.0)
    extracurricular_participation: int = Field(..., ge=0,   le=1,   example=1,   description="0=No 1=Yes")


class PredictionResponse(BaseModel):
    predicted_exam_score: float
    model_type: str
    model_source: str


FEATURE_ORDER = [
    "age", "gender", "study_hours_per_day", "social_media_hours", "netflix_hours",
    "part_time_job", "attendance_percentage", "sleep_hours", "diet_quality",
    "exercise_frequency", "parental_education_level", "internet_quality",
    "mental_health_rating", "extracurricular_participation",
]


def features_to_scaled_array(features: StudentFeatures) -> np.ndarray:
    row = pd.DataFrame([features.dict()])[FEATURE_ORDER]
    return scaler.transform(row)


app = FastAPI(
    title="Student Habits → Exam Score Predictor",
    description="Run train_pipeline.py first to generate the required artefacts.",
    version="1.0.0",
)


@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "model_type": type(model).__name__, "model_source": model_source}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: StudentFeatures):
    try:
        X = features_to_scaled_array(features)
        score = float(model.predict(X)[0])
        score = max(0.0, min(100.0, score))
        return PredictionResponse(
            predicted_exam_score=round(score, 2),
            model_type=type(model).__name__,
            model_source=model_source,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "model_source": model_source,
        "model_params": model.get_params(),
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_artifact_root": MLFLOW_ARTIFACT_ROOT,
        "project_dir": str(PROJECT_DIR),
    }
