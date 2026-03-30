from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "models"


app = FastAPI(title="Student Habits Prediction API")


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=100)
    gender: str
    study_hours_per_day: float = Field(..., ge=0)
    social_media_hours: float = Field(..., ge=0)
    netflix_hours: float = Field(..., ge=0)
    part_time_job: str
    attendance_percentage: float = Field(..., ge=0, le=100)
    sleep_hours: float = Field(..., ge=0, le=24)
    diet_quality: str
    exercise_frequency: int = Field(..., ge=0)
    parental_education_level: str
    internet_quality: str
    mental_health_rating: int = Field(..., ge=0, le=10)
    extracurricular_participation: str

    @field_validator(
        "gender",
        "part_time_job",
        "diet_quality",
        "parental_education_level",
        "internet_quality",
        "extracurricular_participation",
        mode="before",
    )
    @classmethod
    def clean_text(cls, value):
        if not isinstance(value, str):
            raise ValueError("This field must be text")
        return value.strip().title()


def load_model():
    """ Load the saved best model. """
    
    model_path = MODELS_DIR / "best_model.joblib"
    return joblib.load(model_path)


def load_feature_columns():
    """ Load feature column names. """
    
    features_path = MODELS_DIR / "feature_columns.joblib"
    return joblib.load(features_path)


def load_model_info():
    """ Load saved model info. """
    
    info_path = MODELS_DIR / "best_model_info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_category_maps():
    """
    Load saved category mappings.
    """
    maps_path = MODELS_DIR / "category_maps.json"
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_raw_dataframe(df: pd.DataFrame, category_maps):
    """
    Prepare raw dataset rows like training.
    """
    df = df.copy()

    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    if "parental_education_level" in df.columns:
        df["parental_education_level"] = df["parental_education_level"].fillna(
            df["parental_education_level"].mode()[0]
        )

    for column, mapping in category_maps.items():
        df[column] = df[column].astype(str).str.strip().str.title()
        df[column] = df[column].map(mapping)

    return df


def prepare_input_row(request: PredictionRequest, category_maps, feature_columns):
    """
    Convert API input into the same numeric format used in training.
    """
    row = request.model_dump()

    for column, mapping in category_maps.items():
        value = row[column]
        if value not in mapping:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value for {column}. Allowed values: {list(mapping.keys())}"
            )
        row[column] = mapping[value]

    df = pd.DataFrame([row])[feature_columns]
    return df


model = load_model()
feature_columns = load_feature_columns()
model_info = load_model_info()
category_maps = load_category_maps()


@app.get("/")
def home():
    return {
        "message": "Student Habits Prediction API is running",
        "best_model_name": model_info["best_model_name"],
        "number_of_features": len(feature_columns)
    }


@app.get("/predict-sample/{sample_index}")
def predict_sample(sample_index: int):
    """
    Predict one sample directly from the dataset by index.
    Good for testing and demo.
    """
    df = pd.read_csv(DATA_PATH)
    df = prepare_raw_dataframe(df, category_maps)

    if sample_index < 0 or sample_index >= len(df):
        raise HTTPException(
            status_code=400,
            detail=f"sample_index must be between 0 and {len(df) - 1}"
        )

    X_sample = df.loc[[sample_index], feature_columns]
    actual_score = float(df.loc[sample_index, "exam_score"])
    predicted_score = float(model.predict(X_sample)[0])

    return {
        "sample_index": sample_index,
        "actual_exam_score": round(actual_score, 2),
        "predicted_exam_score": round(predicted_score, 2)
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Predict from manually provided values.
    """
    X_input = prepare_input_row(request, category_maps, feature_columns)
    predicted_score = float(model.predict(X_input)[0])
    predicted_score = max(0.0, min(100.0, predicted_score))

    return {
        "predicted_exam_score": round(predicted_score, 2),
        "best_model_name": model_info["best_model_name"]
    }