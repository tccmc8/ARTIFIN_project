import json
import os
import joblib
import pandas as pd
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------- Constants ----------
TARGET_COLUMN = "exam_score"

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

    # Drop identifier — no predictive value
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    # Fill missing parental education values before encoding
    if "parental_education_level" in df.columns:
        df["parental_education_level"] = df["parental_education_level"].fillna(
            df["parental_education_level"].mode()[0]
        )

    # Encode categorical columns to integers
    for column, mapping in category_maps.items():
        df[column] = df[column].astype(str).str.strip().str.title()
        df[column] = df[column].map(mapping)

    return df


def main():
    os.makedirs(PROJECT_ROOT / "data", exist_ok=True)

    print("Loading model artifacts...")
    model = load_model()
    feature_columns = load_feature_columns()
    category_maps = load_category_maps()

    print("Loading raw dataset...")
    df = pd.read_csv(DATA_PATH)

    raw_numeric = df[NUMERIC_FEATURES].copy()

    print("Preprocessing dataset...")
    df_encoded = prepare_dataframe(df, category_maps)

    # Generate predictions on the full dataset
    X = df_encoded[feature_columns]
    predictions = model.predict(X)

    # Build the reference dataset:
    reference = raw_numeric.copy()
    reference[TARGET_COLUMN] = df_encoded[TARGET_COLUMN].values
    reference["prediction"] = predictions.round(2)

    out_path = PROJECT_ROOT / "data" / "reference.csv"
    reference.to_csv(out_path, index=False)
    print(f"reference.csv created at {out_path}  ({len(reference)} rows)")


if __name__ == "__main__":
    main()