import os
import glob
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2 as psycopg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference.csv"
BATCHES_DIR = PROJECT_ROOT / "data" / "current_batches"


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


def main():
    # Load data
    reference = pd.read_csv(REFERENCE_PATH)

    batch_files = glob.glob(str(BATCHES_DIR / "*.csv"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {BATCHES_DIR}")

    # Always evaluate the most recently created batch file
    latest_file = max(batch_files, key=os.path.getmtime)
    current = pd.read_csv(latest_file)
    batch_id = os.path.basename(latest_file).replace(".csv", "")

    print(f"Evaluating batch: {batch_id}  ({len(current)} rows)")

    # Drift detection 
    drifted_features = []
    for feature in NUMERIC_FEATURES:
        _, p_value = ks_2samp(reference[feature], current[feature])
        if p_value < 0.05:
            drifted_features.append(feature)

    num_drifted = len(drifted_features)
    share_drifted = num_drifted / len(NUMERIC_FEATURES)

    if drifted_features:
        print(f"Drifted features ({num_drifted}): {drifted_features}")
    else:
        print("No feature drift detected.")

    # Regression performance
    y_true = current["exam_score"]
    y_pred = current["prediction"]

    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    # Prediction distribution
    pred_mean = float(y_pred.mean())
    pred_std = float(y_pred.std())
    pred_min = float(y_pred.min())
    pred_max = float(y_pred.max())

    # Put to Postgres
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "example"),
    )

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                timestamp                   TIMESTAMP,
                batch_id                    TEXT,
                batch_size                  INT,
                num_drifted_features        INT,
                share_drifted_features      FLOAT,
                drifted_feature_names       TEXT,
                rmse                        FLOAT,
                mae                         FLOAT,
                r2                          FLOAT,
                pred_mean                   FLOAT,
                pred_std                    FLOAT,
                pred_min                    FLOAT,
                pred_max                    FLOAT
            );
        """)

        cur.execute("""
            INSERT INTO metrics (
                timestamp,
                batch_id,
                batch_size,
                num_drifted_features,
                share_drifted_features,
                drifted_feature_names,
                rmse,
                mae,
                r2,
                pred_mean,
                pred_std,
                pred_min,
                pred_max
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            datetime.utcnow(),
            batch_id,
            len(current),
            num_drifted,
            share_drifted,
            ", ".join(drifted_features) if drifted_features else None,
            rmse,
            mae,
            r2,
            pred_mean,
            pred_std,
            pred_min,
            pred_max,
        ))

    conn.commit()
    conn.close()

    print("Metrics saved to database.")


if __name__ == "__main__":
    main()