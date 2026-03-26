# train_pipeline.py
# Trains two regression models (Ridge + Random Forest) on the Student Habits
# vs Academic Performance dataset, tracks every experiment with MLflow, and
# registers the best model in the MLflow Model Registry.
#
# Usage:
#   export MLFLOW_TRACKING_URI=http://localhost:5000   # or file:./mlruns
#   python train_pipeline.py

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Constants ────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Student_Habits_Academic_Performance"
MLRUNS_DIR      = "./mlruns"
CV_FOLDS        = 5
RANDOM_STATE    = 42
TARGET          = "exam_score"
DROP_COLS       = ["student_id", TARGET]


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Download the dataset from Kaggle and return a DataFrame."""
    kagglehub.dataset_download(
        "jayaantanaath/student-habits-vs-academic-performance"
    )
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jayaantanaath/student-habits-vs-academic-performance",
        "student_habits_performance.csv",
    )
    print(f"[load_data] shape={df.shape}")
    return df


# ── 2. Preprocessing ──────────────────────────────────────────────────────────
def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Print and return missing-value counts per column."""
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0])
    return missing


def fill_missing_education(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN parental_education_level with the column mode."""
    df["parental_education_level"] = df["parental_education_level"].fillna(
        df["parental_education_level"].mode()[0]
    )
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode all categorical columns used as features."""
    mappings = {
        "gender": {"Male": 0, "Female": 1, "Other": 2},
        "diet_quality": {"Poor": 0, "Fair": 1, "Good": 2},
        "parental_education_level": {"High School": 0, "Bachelor": 1, "Master": 2},
        "internet_quality": {"Poor": 0, "Average": 1, "Good": 2},
        "extracurricular_participation": {"No": 0, "Yes": 1},
        "part_time_job": {"No": 0, "Yes": 1},
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].str.strip().str.title().map(mapping)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run all preprocessing steps in order."""
    check_missing_values(df)
    df = fill_missing_education(df)
    df = encode_categoricals(df)
    # Drop rows where the target is NaN (safety net)
    df = df.dropna(subset=[TARGET])
    return df


# ── 3. Train / val / test split ───────────────────────────────────────────────
def train_val_test_split(df: pd.DataFrame):
    """60 / 20 / 20 stratified split on a binned version of the target."""
    X = df.drop(columns=DROP_COLS, errors="ignore")
    y = df[TARGET]

    # Bin target for stratify (ensures balanced splits across score ranges)
    y_binned = pd.cut(y, bins=5, labels=False)

    X_train, X_temp, y_train, y_temp, yb_train, yb_temp = train_test_split(
        X, y, y_binned, test_size=0.40, random_state=RANDOM_STATE, stratify=y_binned
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=yb_temp
    )
    print(
        f"[split] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 4. Scaler ─────────────────────────────────────────────────────────────────
def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit and save a StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, "scaler.joblib")
    print("[fit_scaler] scaler saved to scaler.joblib")
    return scaler


# ── 5. Metrics helper ─────────────────────────────────────────────────────────
def eval_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }


# ── 6. MLflow experiment runner ───────────────────────────────────────────────
def train_and_log_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train Ridge Regression and Random Forest Regressor.
    Each model gets its own MLflow run.  The model with the lower test RMSE
    is registered as 'BestStudentHabitsModel'.
    """
    os.makedirs(MLRUNS_DIR, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models_cfg = [
        {
            "label": "Ridge Regression",
            "model": Ridge(alpha=1.0, random_state=RANDOM_STATE),
            "params": {"alpha": 1.0},
        },
        {
            "label": "Random Forest",
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 4,
            },
        },
    ]

    results = {}  # label → {"run_id", "test_rmse", "model"}

    for cfg in models_cfg:
        label = cfg["label"]
        model = cfg["model"]
        print(f"\n[mlflow] Starting run: {label}")

        with mlflow.start_run(run_name=label) as run:
            # ── Cross-validation on training fold ────────────────────────
            cv_rmse = -cross_val_score(
                model, X_train, y_train,
                cv=kfold, scoring="neg_root_mean_squared_error"
            )

            # ── Fit on full training fold ────────────────────────────────
            model.fit(X_train, y_train)

            # ── Evaluate on val & test ───────────────────────────────────
            val_m  = eval_metrics(y_val,  model.predict(X_val))
            test_m = eval_metrics(y_test, model.predict(X_test))

            # ── Log params ───────────────────────────────────────────────
            mlflow.log_param("model_type",   label)
            mlflow.log_param("cv_folds",     CV_FOLDS)
            mlflow.log_param("random_state", RANDOM_STATE)
            for k, v in cfg["params"].items():
                mlflow.log_param(k, v)

            # ── Log metrics ──────────────────────────────────────────────
            mlflow.log_metric("cv_rmse_mean", float(cv_rmse.mean()))
            mlflow.log_metric("cv_rmse_std",  float(cv_rmse.std()))
            for split, m in [("val", val_m), ("test", test_m)]:
                for metric_name, value in m.items():
                    mlflow.log_metric(f"{split}_{metric_name}", value)

            # ── Log model ────────────────────────────────────────────────
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=None,   # register only the best later
            )

            run_id = run.info.run_id
            results[label] = {
                "run_id": run_id,
                "test_rmse": test_m["rmse"],
                "model": model,
            }

            print(
                f"  cv_rmse={cv_rmse.mean():.4f}±{cv_rmse.std():.4f} | "
                f"val_r2={val_m['r2']:.4f} | "
                f"test_rmse={test_m['rmse']:.4f}  run_id={run_id}"
            )

    # ── Register the best model ───────────────────────────────────────────────
    best_label = min(results, key=lambda k: results[k]["test_rmse"])
    best_run_id = results[best_label]["run_id"]
    print(f"\n[mlflow] Best model: {best_label} (run_id={best_run_id})")

    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri, "BestStudentHabitsModel")
    print("[mlflow] Model registered as 'BestStudentHabitsModel'")

    # Also save best model locally for the FastAPI service
    best_model = results[best_label]["model"]
    joblib.dump(best_model, "best_model.joblib")
    print("[mlflow] Best model saved to best_model.joblib")

    return results


# ── 7. Main ───────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    df = preprocess(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    scaler = fit_scaler(X_train)

    # Scale splits
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    train_and_log_models(
        X_train_s, X_val_s, X_test_s,
        y_train,   y_val,   y_test
    )


if __name__ == "__main__":
    main()
