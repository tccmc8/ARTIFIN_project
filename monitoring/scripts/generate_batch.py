import json
import os
import uuid
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BATCHES_DIR = PROJECT_ROOT / "data" / "current_batches"

TARGET_COLUMN = "exam_score"
BATCH_SIZE = 50

NUMERIC_FEATURES = [
    "age",
    "study_hours_per_day",
    "social_media_hours",
    "netflix_hours",
    "attendance_percentage",
    "sleep_hours",
    "exercise_frequency",
    "mental_health_rating",
]


def load_model():
    model_path = MODELS_DIR / "best_model.joblib"
    return joblib.load(model_path)


def load_feature_columns():
    features_path = MODELS_DIR / "feature_columns.joblib"
    return joblib.load(features_path)


def load_category_maps():
    maps_path = MODELS_DIR / "category_maps.json"
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataframe(df: pd.DataFrame, category_maps: dict) -> pd.DataFrame:
    """Apply the same preprocessing pipeline used during training."""
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


#def introduce_drift(df_raw: pd.DataFrame) -> pd.DataFrame:
    """ Artificial dift:
    Here we increase study_hours_per_day by 2 hours and reduce
    social_media_hours by 1 hour, mimicking a student behaviour change.
    """
    #df = df_raw.copy()

    # uncomment to simulate artificial drift
    # df["study_hours_per_day"] = (df["study_hours_per_day"] + 2.0).clip(upper=24)
    # df["social_media_hours"] = (df["social_media_hours"] - 1.0).clip(lower=0)

    #return df


def main():
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model artifacts...")
    model = load_model()
    feature_columns = load_feature_columns()
    category_maps = load_category_maps()

    print("Loading raw dataset...")
    df = pd.read_csv(DATA_PATH)

    batch_raw = df.sample(n=BATCH_SIZE, replace=True, random_state=None).reset_index(drop=True)

    #batch_raw = introduce_drift(batch_raw)

    batch_numeric = batch_raw[NUMERIC_FEATURES].copy()

    # Encode and predict
    batch_encoded = prepare_dataframe(batch_raw, category_maps)
    X_batch = batch_encoded[feature_columns]
    predictions = model.predict(X_batch)

    # Build the output batch:
    batch_out = batch_numeric.copy()
    batch_out[TARGET_COLUMN] = batch_encoded[TARGET_COLUMN].values
    batch_out["prediction"] = predictions.round(2)

    batch_id = str(uuid.uuid4())[:8]
    out_path = BATCHES_DIR / f"{batch_id}.csv"
    batch_out.to_csv(out_path, index=False)

    print(f"batch saved: {batch_id}.csv  ({len(batch_out)} rows)")


if __name__ == "__main__":
    main()