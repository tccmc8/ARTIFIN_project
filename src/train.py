from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "./models/"
MLRUNS_DIR = PROJECT_ROOT / "./mlruns/"

# ---------- Constants ----------
TARGET_COLUMN = "exam_score"
EXPERIMENT_NAME = "student-habits-performance"


def set_mlflow_tracking():
    """ This creates an mlruns folder in the project root if there isn't one already, 
    setting the tracking UI to this folder. A set folder to store the experiment data. 
    It worksregarless of the working directory. """

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())


def load_data(path: Path) -> pd.DataFrame:
    """ Load the dataset CSV."""

    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    """
    Cleans the dataset of missing values, then encodes text columns into numbers. Then 
    splits the dataset into features and target - the x, y split. """

    df = df.copy()

    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    if "parental_education_level" in df.columns:
        df["parental_education_level"] = df["parental_education_level"].fillna(
            df["parental_education_level"].mode()[0]
        )

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

    return X, y, feature_columns, category_maps


def split_data(X: pd.DataFrame, y: pd.Series):
    """ Split data into an 80/20 training and test sets. """

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


def build_models():
    """
    Builds the two two regression models. """
    models = {
        "RidgeRegression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),
            "params": {
                "model_type": "RidgeRegression",
                "alpha": 1.0
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }
    }
    return models


def evaluate_model(y_true, y_pred):
    """ Calculate metrics for the two models. """
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }
    return metrics


def train_and_log_models(X_train, X_test, y_train, y_test):
    """ Creates an MLflow experiment and sets up the measuring assignments. Runs / trains
    both models and logs them to MLflow, including the parameters. Performs the model 
    training and testing and logs the metrics in MLflow. It compares the models using an if
    loop and returns the best model, its name, its metrics, and a dataframe with all the 
    results. """

    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = []
    best_model = None
    best_model_name = None
    best_metrics = None
    best_score = float("inf")

    models = build_models()

    for model_name, model_info in models.items():
        model = model_info["model"]
        params = model_info["params"]

        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = evaluate_model(y_test, y_pred)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, artifact_path="model")

            result_row = {
                "model_name": model_name,
                **metrics
            }
            all_results.append(result_row)

            if metrics["rmse"] < best_score:
                best_score = metrics["rmse"]
                best_model = model
                best_model_name = model_name
                best_metrics = metrics

    results_df = pd.DataFrame(all_results).sort_values(
        by="rmse",
        ascending=True
    )

    return best_model, best_model_name, best_metrics, results_df


def save_best_model(best_model, best_model_name, best_metrics, feature_columns, category_maps):
    """ Creates readable files that show the model metrics and name and a file that shows
    the category maps. The eperiment makes two runs, one for each model and then saves the 
    best model as joblib locally. """

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

    return model_path, features_path, info_path, maps_path


def main():
    print("Setting MLflow tracking...")
    set_mlflow_tracking()

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preparing data...")
    X, y, feature_columns, category_maps = prepare_data(df)

    print("Number of input features:", len(feature_columns))

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training models and logging to MLflow...")
    best_model, best_model_name, best_metrics, results_df = train_and_log_models(
        X_train, X_test, y_train, y_test
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    print("\nSaving best model locally...")
    model_path, features_path, info_path, maps_path = save_best_model(
        best_model,
        best_model_name,
        best_metrics,
        feature_columns,
        category_maps
    )

    print("\nBest model selected:", best_model_name)
    print("Best metrics:", best_metrics)
    print("Saved model to:", model_path)
    print("Saved feature columns to:", features_path)
    print("Saved model info to:", info_path)
    print("Saved category maps to:", maps_path)

    print("\nTo open MLflow UI, run:")
    print(f"mlflow ui --backend-store-uri {MLRUNS_DIR.as_uri()} --port 5000")

    print("\nDone.")


if __name__ == "__main__":
    main()