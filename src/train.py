from pathlib import Path
import json
import os
 
# MLflow 3.x deprecates the on-disk file store by default; opt back in so the
# existing mlruns/ directory keeps working without forcing a DB migration.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
 
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
 
from xgboost import XGBRegressor


# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "./models/"
MLRUNS_DIR = PROJECT_ROOT / "./mlruns/"

# ---------- Constants ----------
TARGET_COLUMN = "exam_score"
EXPERIMENT_NAME = "student-habits-performance"
RANDOM_STATE = 42
CV_FOLDS = 5


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

    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def build_models():
    """ Builds the five regression models with their hyperparameter grids. """

    models = {

        "RidgeRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
        },
 
        "Lasso": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=10000, random_state=RANDOM_STATE)),
            ]),
            "param_grid": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
            },
        },

        "KNN": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_jobs=-1)),
            ]),
            "param_grid": {
                "model__n_neighbors": [5, 10, 15, 20],
                "model__weights": ["uniform", "distance"],
            },
        },
 
        "RandomForest": {
            "pipeline": Pipeline([
                ("model", RandomForestRegressor(
                    random_state=RANDOM_STATE, n_jobs=-1
                )),
            ]),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, 20],
                "model__min_samples_split": [2, 5],
            },
        },
 
        "XGBoost": {
            "pipeline": Pipeline([
                ("model", XGBRegressor(
                    random_state=RANDOM_STATE,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    verbosity=0,
                )),
            ]),
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
            },
        },
    }

    return models


def evaluate_model(y_true, y_pred):
    """ Calculate metrics for the models. """
    metrics = {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }
    return metrics


def tune_and_evaluate(name, spec, X_train, X_test, y_train, y_test, cv):
    """ Grid-search one model, log everything to MLflow, return the fitted best
    estimator along with its test-set metrics and chosen hyperparameters."""

    grid = GridSearchCV(
        estimator=spec["pipeline"],
        param_grid=spec["param_grid"],
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
 
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_param("cv_folds", cv.get_n_splits())
        mlflow.log_param("param_grid", json.dumps(spec["param_grid"]))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
 
        grid.fit(X_train, y_train)
 
        cv_best_rmse = float(-grid.best_score_)
        mlflow.log_metric("cv_best_rmse", cv_best_rmse)
        for param, value in grid.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
 
        y_pred = grid.best_estimator_.predict(X_test)
        test_metrics = evaluate_model(y_test, y_pred)
        mlflow.log_metrics(test_metrics)
 
        mlflow.sklearn.log_model(grid.best_estimator_, artifact_path="model")
 
    return grid.best_estimator_, test_metrics, grid.best_params_, cv_best_rmse


def build_voting_ensemble(tuned_estimators):
    """Soft-vote regressor (simple unweighted average of base predictions) built
    from already-tuned base estimators. Each entry is (name, fitted_pipeline)."""

    return VotingRegressor(estimators=tuned_estimators, n_jobs=-1)


def train_and_log_models(X_train, X_test, y_train, y_test):
    """ Creates an MLflow experiment and sets up the measuring assignments. Runs / trains
    both models and logs them to MLflow, including the parameters. Performs the model 
    training and testing and logs the metrics in MLflow. It compares the models using an if
    loop and returns the best model, its name, its metrics, and a dataframe with all the 
    results. """

    mlflow.set_experiment(EXPERIMENT_NAME)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
 
    all_results = []
    tuned = {}        # name -> fitted best estimator
 
    best_model = None
    best_model_name = None
    best_metrics = None
    best_score = float("inf")
 
    # ---- 1. Tune every base model ----
    specs = build_models()
    for name, spec in specs.items():
        print(f"\n  Tuning {name}...")
        est, metrics, best_params, cv_best = tune_and_evaluate(
            name, spec, X_train, X_test, y_train, y_test, cv
        )
        tuned[name] = est
 
        all_results.append({
            "model_name":  name,
            "best_params": best_params,
            "cv_rmse":     round(cv_best, 4),
            **{k: round(v, 4) for k, v in metrics.items()},
        })
 
        if metrics["rmse"] < best_score:
            best_score = metrics["rmse"]
            best_model = est
            best_model_name = name
            best_metrics = metrics
 
    # ---- 2. Build and evaluate the voting ensemble ----
    print("\n  Building VotingRegressor from tuned base models...")
    ensemble_members = [(n.lower(), tuned[n]) for n in tuned]
    voting = build_voting_ensemble(ensemble_members)
 
    with mlflow.start_run(run_name="VotingRegressor"):
        mlflow.log_param("model_type", "VotingRegressor")
        mlflow.log_param("members", ",".join(tuned.keys()))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
 
        voting.fit(X_train, y_train)
        y_pred = voting.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(voting, artifact_path="model")
 
    all_results.append({
        "model_name":  "VotingRegressor",
        "best_params": {"members": list(tuned.keys())},
        "cv_rmse":     np.nan,
        **{k: round(v, 4) for k, v in metrics.items()},
    })
 
    if metrics["rmse"] < best_score:
        best_score = metrics["rmse"]
        best_model = voting
        best_model_name = "VotingRegressor"
        best_metrics = metrics
 
    results_df = pd.DataFrame(all_results).sort_values(by="rmse", ascending=True)

    return best_model, best_model_name, best_metrics, results_df


def save_best_model(best_model, best_model_name, best_metrics, feature_columns, category_maps):
    """ Creates readable files that show the model metrics and name and a file that shows
    the category maps. The eperiment makes two runs, one for each model and then saves the 
    best model as joblib locally. """

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
 
    model_path    = MODELS_DIR / "best_model.joblib"
    features_path = MODELS_DIR / "feature_columns.joblib"
    info_path     = MODELS_DIR / "best_model_info.json"
    maps_path     = MODELS_DIR / "category_maps.json"
 
    joblib.dump(best_model, model_path)
    joblib.dump(feature_columns, features_path)
 
    info = {
        "best_model_name": best_model_name,
        "selection_metric": "rmse",
        "metrics": best_metrics,
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
 
    print("Tuning and training models, logging to MLflow...")
    best_model, best_model_name, best_metrics, results_df = train_and_log_models(
        X_train, X_test, y_train, y_test
    )
 
    print("\nModel comparison (sorted by test RMSE, lower is better):")
    print(results_df.to_string(index=False))
 
    print("\nSaving best model locally...")
    model_path, features_path, info_path, maps_path = save_best_model(
        best_model, best_model_name, best_metrics, feature_columns, category_maps
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
