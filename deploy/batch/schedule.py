from pathlib import Path
import json
import os
import tempfile

import joblib
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd

from prefect import flow, task, get_run_logger

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# ---------- Constants ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
EXPERIMENT_NAME = "student-habits-performance"
TARGET_COLUMN = "exam_score"


# ──────────────────────────────────────────
# TASKS
# ──────────────────────────────────────────

@task
def load_data() -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    logger = get_run_logger()
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


@task
def check_data_quality(df: pd.DataFrame) -> pd.Series:
    """
    Log a summary of missing values across all columns.
    Helps catch data problems early before training begins.
    """
    logger = get_run_logger()
    missing = df.isnull().sum()
    logger.info(f"Missing values per column:\n{missing}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Target column stats:\n{df[TARGET_COLUMN].describe()}")
    return missing


@task
def prepare_data(df: pd.DataFrame):
    """
    Clean the dataset and encode categorical text columns into numbers.
    Returns the feature matrix X, target vector y,
    the list of feature column names, and the category mapping dictionary.
    """
    logger = get_run_logger()
    df = df.copy()

    # Drop student_id — it is a unique identifier with no predictive value
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    # Fill missing parental education values with the most common value
    if "parental_education_level" in df.columns:
        df["parental_education_level"] = df["parental_education_level"].fillna(
            df["parental_education_level"].mode()[0]
        )

    # Map text categories to integers so the model can work with them
    category_maps = {
        "gender": {"Male": 0, "Female": 1, "Other": 2},
        "part_time_job": {"No": 0, "Yes": 1},
        "diet_quality": {"Poor": 0, "Fair": 1, "Good": 2},
        "parental_education_level": {"High School": 0, "Bachelor": 1, "Master": 2},
        "internet_quality": {"Poor": 0, "Average": 1, "Good": 2},
        "extracurricular_participation": {"No": 0, "Yes": 1},
    }

    for column, mapping in category_maps.items():
        df[column] = df[column].astype(str).str.strip().str.title()
        df[column] = df[column].map(mapping)

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    logger.info(f"Number of features: {len(feature_columns)}")
    logger.info(f"Feature columns: {feature_columns}")

    return X, y, feature_columns, category_maps


@task
def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Split data into training (80%) and test (20%) sets.
    random_state=42 ensures the split is identical every run.
    """
    logger = get_run_logger()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test


@task
def train_and_log_models(X_train, X_test, y_train, y_test, feature_columns, category_maps):
    """
    Train both Ridge Regression and Random Forest models.
    Log each run to MLflow with its parameters, metrics, and model artefact.
    Compare on RMSE and return the best model along with its run ID.
    """
    logger = get_run_logger()

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Define both models and their hyperparameters
    models = {
        "RidgeRegression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),
            "params": {"model_type": "RidgeRegression", "alpha": 1.0}
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "params": {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }
    }

    best_model = None
    best_model_name = None
    best_metrics = None
    best_run_id = None
    best_score = float("inf")
    all_results = []

    for model_name, model_info in models.items():
        model = model_info["model"]
        params = model_info["params"]

        with mlflow.start_run(run_name=model_name) as run:
            # Log hyperparameters
            mlflow.log_params(params)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            # Train and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate regression metrics
            metrics = {
                "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred))
            }

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Save the category maps as an artefact so they travel with the model
            with tempfile.TemporaryDirectory() as tmpdir:
                maps_path = Path(tmpdir) / "category_maps.json"
                with open(maps_path, "w", encoding="utf-8") as f:
                    json.dump(category_maps, f, indent=4)
                mlflow.log_artifact(str(maps_path), artifact_path="preprocessing")

                features_path = Path(tmpdir) / "feature_columns.joblib"
                joblib.dump(feature_columns, features_path)
                mlflow.log_artifact(str(features_path), artifact_path="preprocessing")

            run_id = run.info.run_id
            logger.info(f"{model_name} | RMSE: {metrics['rmse']:.4f} | R2: {metrics['r2']:.4f} | run_id: {run_id}")

            all_results.append({"model_name": model_name, **metrics})

            # Keep track of the best model based on lowest RMSE
            if metrics["rmse"] < best_score:
                best_score = metrics["rmse"]
                best_model = model
                best_model_name = model_name
                best_metrics = metrics
                best_run_id = run_id

    results_df = pd.DataFrame(all_results).sort_values(by="rmse", ascending=True)
    logger.info(f"\nModel comparison:\n{results_df.to_string(index=False)}")
    logger.info(f"Best model: {best_model_name} with RMSE {best_score:.4f}")

    return best_model, best_model_name, best_metrics, best_run_id


@task
def save_best_model(best_model, best_model_name, best_metrics, feature_columns, category_maps):
    """
    Save the best model and all supporting files to the local models/ folder
    so the FastAPI app can load them at startup.
    """
    logger = get_run_logger()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    features_path = MODELS_DIR / "feature_columns.joblib"
    info_path = MODELS_DIR / "best_model_info.json"
    maps_path = MODELS_DIR / "category_maps.json"

    joblib.dump(best_model, model_path)
    joblib.dump(feature_columns, features_path)

    info = {
        "best_model_name": best_model_name,
        "selection_metric": "rmse",
        "metrics": best_metrics
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump(category_maps, f, indent=4)

    logger.info(f"Saved best model ({best_model_name}) to {model_path}")
    return model_path


@task
def generate_batch(df: pd.DataFrame, n_samples: int = 100) -> pd.DataFrame:
    """
    Sample n_samples rows from the dataset to use as a batch prediction test.
    replace=True allows the same row to be picked more than once.
    """
    logger = get_run_logger()
    features = df.drop(columns=["student_id", TARGET_COLUMN], errors="ignore")
    batch = features.sample(n=n_samples, replace=True, random_state=42).reset_index(drop=True)
    logger.info(f"Generated batch of {len(batch)} samples")
    return batch


@task
def batch_predict(run_id: str, batch_df: pd.DataFrame):
    """
    Load the best model directly from MLflow using its run ID,
    run predictions on the batch, and save results to outputs/.
    """
    logger = get_run_logger()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load model artefact directly from MLflow — no need to touch the local models/ folder
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

    # Download the category maps and feature columns that were saved alongside the model
    preprocessing_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/preprocessing"
    )
    preprocessing_path = Path(preprocessing_dir)

    category_maps = json.loads((preprocessing_path / "category_maps.json").read_text())
    feature_columns = joblib.load(preprocessing_path / "feature_columns.joblib")

    # Apply the same encoding used during training
    batch_encoded = batch_df.copy()
    for column, mapping in category_maps.items():
        if column in batch_encoded.columns:
            batch_encoded[column] = batch_encoded[column].astype(str).str.strip().str.title()
            batch_encoded[column] = batch_encoded[column].map(mapping)

    X_batch = batch_encoded[feature_columns]
    predictions = model.predict(X_batch)

    result = batch_df.copy()
    result["predicted_exam_score"] = predictions.round(2)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"batch_predictions_{run_id}.csv"
    result.to_csv(out_file, index=False)

    logger.info(f"Saved {len(result)} batch predictions to {out_file}")
    return result


# ──────────────────────────────────────────
# FLOW — orchestrates all tasks in order
# ──────────────────────────────────────────

@flow(name="student_habits_train_and_batch_predict")
def student_habits_train_and_batch_predict(n_samples: int = 100) -> str:
    """
    Full end-to-end pipeline:
    1. Load data
    2. Check data quality
    3. Prepare and encode features
    4. Split into train/test
    5. Train both models, log to MLflow, pick the best
    6. Save best model locally for the FastAPI app
    7. Generate a batch of samples
    8. Run batch predictions using the MLflow-registered model
    """
    df = load_data()
    check_data_quality(df)
    X, y, feature_columns, category_maps = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    best_model, best_model_name, best_metrics, best_run_id = train_and_log_models(
        X_train, X_test, y_train, y_test, feature_columns, category_maps
    )
    save_best_model(best_model, best_model_name, best_metrics, feature_columns, category_maps)
    batch = generate_batch(df, n_samples=n_samples)
    batch_predict(best_run_id, batch)
    return best_run_id


if __name__ == "__main__":
    student_habits_train_and_batch_predict()