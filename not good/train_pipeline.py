# train_pipeline.py
# Mirrors the StudentHabits_MLflow_Experiments.ipynb exactly.
#
# Two separate MLflow experiments are run:
#   1. Student_Habits_Ridge_Regression  — sweeps alphas [0.1, 1.0, 10.0, 100.0]
#   2. Student_Habits_Random_Forest     — sweeps n_estimators [50, 100, 200, 300]
#
# The best run from each experiment is found via mlflow.search_runs(), both are
# registered, then the overall best is registered as 'BestStudentHabitsModel',
# transitioned to Staging, and saved as best_model.joblib for the FastAPI service.
#
# Usage:
#   python train_pipeline.py

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import kagglehub
from kagglehub import KaggleDatasetAdapter
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── 1. Load data ──────────────────────────────────────────────────────────────
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
    print("Shape:", df.shape)
    return df


# ── 2. Check missing values ───────────────────────────────────────────────────
def check_missing_values(df: pd.DataFrame) -> None:
    """Print missing-value counts per column."""
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    total_missing = df.isnull().sum().sum()
    print(f"\nTotal missing values: {total_missing}")
    df.info()


# ── 3. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values, ordinal-encode categoricals, and drop student_id.
    Matches the notebook's preprocessing cells exactly.
    """
    # Fill missing parental_education_level with the column mode
    df["parental_education_level"] = df["parental_education_level"].fillna(
        df["parental_education_level"].mode()[0]
    )
    print("Missing values after fill:")
    print(df.isnull().sum())

    # Ordinal-encode categorical columns
    gender_map   = {"Male": 0, "Female": 1, "Other": 2}
    diet_map     = {"Poor": 0, "Fair": 1, "Good": 2}
    par_ed_map   = {"High School": 0, "Bachelor": 1, "Master": 2}
    internet_map = {"Poor": 0, "Average": 1, "Good": 2}
    clubs_map    = {"No": 0, "Yes": 1}
    job_map      = {"No": 0, "Yes": 1}

    df["gender"]                       = df["gender"].str.strip().str.title().map(gender_map)
    df["diet_quality"]                  = df["diet_quality"].str.strip().str.title().map(diet_map)
    df["parental_education_level"]      = df["parental_education_level"].str.strip().str.title().map(par_ed_map)
    df["internet_quality"]              = df["internet_quality"].str.strip().str.title().map(internet_map)
    df["extracurricular_participation"] = df["extracurricular_participation"].str.strip().str.title().map(clubs_map)
    df["part_time_job"]                 = df["part_time_job"].str.strip().str.title().map(job_map)

    # Drop student_id — carries no predictive signal
    df = df.drop(columns=["student_id"], errors="ignore")

    return df


# ── 4. Train / validation / test split ───────────────────────────────────────
def train_val_test_split(df: pd.DataFrame):
    """
    60 / 20 / 20 stratified split.
    Matches the notebook's split cell exactly.
    """
    TARGET = "exam_score"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Bin target for stratified splitting
    y_binned = pd.cut(y, bins=5, labels=False)

    # First split: 60 % train, 40 % temp
    X_train, X_temp, y_train, y_temp, yb_train, yb_temp = train_test_split(
        X, y, y_binned, test_size=0.40, random_state=42, stratify=y_binned
    )
    # Second split: 20 % val, 20 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=yb_temp
    )

    print(f"Train set:      {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 5. Scale features ─────────────────────────────────────────────────────────
def fit_and_scale(X_train, X_val, X_test):
    """
    Fit StandardScaler on training data only, save it, and return scaled splits.
    Matches the notebook's scaling cell exactly.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, "scaler.joblib")
    print("Scaler saved to scaler.joblib")

    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_val_s, X_test_s


# ── 6. MLflow setup ───────────────────────────────────────────────────────────
def setup_mlflow() -> None:
    """Create mlruns dir and set the tracking URI. Matches the notebook."""
    os.makedirs("./mlruns", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    print("MLflow tracking URI:", mlflow.get_tracking_uri())


# ── 7. Experiment 1 — Ridge Regression ───────────────────────────────────────
def run_ridge_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test):
    """
    Sweeps alphas [0.1, 1.0, 10.0, 100.0].
    Registers each run, then finds + registers the best.
    Mirrors notebook Section 7 exactly.
    """
    mlflow.set_experiment("Student_Habits_Ridge_Regression")

    alphas = [0.1, 1.0, 10.0, 100.0]

    for alpha in alphas:
        with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):

            # Train
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_s, y_train)

            # Predict
            y_val_pred  = model.predict(X_val_s)
            y_test_pred = model.predict(X_test_s)

            # Metrics
            val_rmse  = float(np.sqrt(mean_squared_error(y_val,  y_val_pred)))
            val_mae   = float(mean_absolute_error(y_val,  y_val_pred))
            val_r2    = float(r2_score(y_val,  y_val_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            test_mae  = float(mean_absolute_error(y_test, y_test_pred))
            test_r2   = float(r2_score(y_test, y_test_pred))

            # Log parameters
            mlflow.log_param("model_type",   "Ridge")
            mlflow.log_param("alpha",        alpha)
            mlflow.log_param("random_state", 42)

            # Log metrics
            mlflow.log_metric("val_rmse",  val_rmse)
            mlflow.log_metric("val_mae",   val_mae)
            mlflow.log_metric("val_r2",    val_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae",  test_mae)
            mlflow.log_metric("test_r2",   test_r2)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id
            print(f"alpha={alpha:<6}  val_r2={val_r2:.4f}  test_rmse={test_rmse:.4f}  run_id={run_id}")

        # Register each run under its own name
        mlflow.register_model(f"runs:/{run_id}/model", f"student_ridge_alpha_{alpha}")

    # Find the best Ridge run (lowest test RMSE)
    ridge_runs = mlflow.search_runs(
        experiment_names=["Student_Habits_Ridge_Regression"],
        order_by=["metrics.test_rmse ASC"]
    )
    best_ridge_run       = ridge_runs.iloc[0]
    best_ridge_run_id    = best_ridge_run["run_id"]
    best_ridge_alpha     = best_ridge_run["params.alpha"]
    best_ridge_test_rmse = best_ridge_run["metrics.test_rmse"]
    best_ridge_test_r2   = best_ridge_run["metrics.test_r2"]

    print(f"\nBest Ridge Model:")
    print(f"  alpha     = {best_ridge_alpha}")
    print(f"  test_rmse = {best_ridge_test_rmse:.4f}")
    print(f"  test_r2   = {best_ridge_test_r2:.4f}")

    mlflow.register_model(f"runs:/{best_ridge_run_id}/model", "best_student_ridge")
    print("Best Ridge model registered as 'best_student_ridge'")

    return best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2


# ── 8. Experiment 2 — Random Forest ──────────────────────────────────────────
def run_rf_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test):
    """
    Sweeps n_estimators [50, 100, 200, 300].
    Registers each run, then finds + registers the best.
    Mirrors notebook Section 8 exactly.
    """
    mlflow.set_experiment("Student_Habits_Random_Forest")

    n_estimators_list = [50, 100, 200, 300]

    for n_est in n_estimators_list:
        with mlflow.start_run(run_name=f"RF_n_estimators_{n_est}"):

            # Train
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_s, y_train)

            # Predict
            y_val_pred  = model.predict(X_val_s)
            y_test_pred = model.predict(X_test_s)

            # Metrics
            val_rmse  = float(np.sqrt(mean_squared_error(y_val,  y_val_pred)))
            val_mae   = float(mean_absolute_error(y_val,  y_val_pred))
            val_r2    = float(r2_score(y_val,  y_val_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            test_mae  = float(mean_absolute_error(y_test, y_test_pred))
            test_r2   = float(r2_score(y_test, y_test_pred))

            # Log parameters
            mlflow.log_param("model_type",       "RandomForest")
            mlflow.log_param("n_estimators",     n_est)
            mlflow.log_param("max_depth",        10)
            mlflow.log_param("min_samples_leaf", 4)
            mlflow.log_param("random_state",     42)

            # Log metrics
            mlflow.log_metric("val_rmse",  val_rmse)
            mlflow.log_metric("val_mae",   val_mae)
            mlflow.log_metric("val_r2",    val_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae",  test_mae)
            mlflow.log_metric("test_r2",   test_r2)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            run_id = mlflow.active_run().info.run_id
            print(f"n_estimators={n_est:<4}  val_r2={val_r2:.4f}  test_rmse={test_rmse:.4f}  run_id={run_id}")

        # Register each run under its own name
        mlflow.register_model(f"runs:/{run_id}/model", f"student_rf_{n_est}_trees")

    # Find the best RF run (lowest test RMSE)
    rf_runs = mlflow.search_runs(
        experiment_names=["Student_Habits_Random_Forest"],
        order_by=["metrics.test_rmse ASC"]
    )
    best_rf_run       = rf_runs.iloc[0]
    best_rf_run_id    = best_rf_run["run_id"]
    best_rf_n_est     = best_rf_run["params.n_estimators"]
    best_rf_test_rmse = best_rf_run["metrics.test_rmse"]
    best_rf_test_r2   = best_rf_run["metrics.test_r2"]

    print(f"\nBest Random Forest Model:")
    print(f"  n_estimators = {best_rf_n_est}")
    print(f"  test_rmse    = {best_rf_test_rmse:.4f}")
    print(f"  test_r2      = {best_rf_test_r2:.4f}")

    mlflow.register_model(f"runs:/{best_rf_run_id}/model", "best_student_rf")
    print("Best Random Forest model registered as 'best_student_rf'")

    return best_rf_run_id, best_rf_n_est, best_rf_test_rmse, best_rf_test_r2


# ── 9. Compare, register overall best, transition to Staging ─────────────────
def register_best_model(
    best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2,
    best_rf_run_id,    best_rf_n_est,    best_rf_test_rmse,    best_rf_test_r2,
    X_test_s, y_test
):
    """
    Compares best Ridge vs best RF on test RMSE, registers the winner as
    'BestStudentHabitsModel', transitions to Staging, and saves
    best_model.joblib for the FastAPI service.
    Mirrors notebook Sections 9-12 exactly.
    """
    print("=" * 50)
    print("Model Comparison — Test Set")
    print("=" * 50)
    print(f"Ridge Regression   test_rmse={best_ridge_test_rmse:.4f}  test_r2={best_ridge_test_r2:.4f}")
    print(f"Random Forest      test_rmse={best_rf_test_rmse:.4f}  test_r2={best_rf_test_r2:.4f}")

    if best_rf_test_rmse < best_ridge_test_rmse:
        overall_best_run_id = best_rf_run_id
        overall_best_label  = f"Random Forest (n_estimators={best_rf_n_est})"
    else:
        overall_best_run_id = best_ridge_run_id
        overall_best_label  = f"Ridge Regression (alpha={best_ridge_alpha})"

    print(f"\n✅ Overall best model: {overall_best_label}")

    # Register overall best
    mlflow.register_model(f"runs:/{overall_best_run_id}/model", "BestStudentHabitsModel")
    print("Overall best model registered as 'BestStudentHabitsModel'")

    # Transition to Staging and add description
    client = MlflowClient()
    date   = datetime.today().strftime("%Y-%m-%d")

    client.transition_model_version_stage(
        name="BestStudentHabitsModel",
        version=1,
        stage="Staging",
        archive_existing_versions=False
    )
    print("Model 'BestStudentHabitsModel' version 1 transitioned to Staging")

    client.update_model_version(
        name="BestStudentHabitsModel",
        version=1,
        description=(
            f"Best model from Student Habits experiments: {overall_best_label}. "
            f"test_rmse={min(best_ridge_test_rmse, best_rf_test_rmse):.4f} as of {date}"
        )
    )
    print("Model description updated.")

    # Load from registry and evaluate (mirrors notebook Section 11)
    model_uri    = "models:/BestStudentHabitsModel/Staging"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    predictions  = loaded_model.predict(X_test_s)

    print("\nFirst 10 predictions vs actual exam scores:")
    for pred, actual in zip(predictions[:10], y_test.values[:10]):
        print(f"  predicted={pred:.2f}   actual={actual:.2f}")

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae  = float(mean_absolute_error(y_test, predictions))
    r2   = float(r2_score(y_test, predictions))
    print(f"\nFinal Test Metrics (loaded from registry):")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    # Save sklearn model locally for the FastAPI service (mirrors notebook Section 12)
    best_sklearn_model = mlflow.sklearn.load_model(f"runs:/{overall_best_run_id}/model")
    joblib.dump(best_sklearn_model, "best_model.joblib")
    print("\nbest_model.joblib saved — ready for predict_pipeline.py")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load
    df = load_data()

    # 2. Inspect
    check_missing_values(df)

    # 3. Preprocess
    df = preprocess(df)

    # 4. Split
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    # 5. Scale
    X_train_s, X_val_s, X_test_s = fit_and_scale(X_train, X_val, X_test)

    # 6. MLflow
    setup_mlflow()

    # 7. Experiment 1 — Ridge
    best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2 = \
        run_ridge_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    # 8. Experiment 2 — Random Forest
    best_rf_run_id, best_rf_n_est, best_rf_test_rmse, best_rf_test_r2 = \
        run_rf_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    # 9. Compare, register best, save artefacts
    register_best_model(
        best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2,
        best_rf_run_id,    best_rf_n_est,    best_rf_test_rmse,    best_rf_test_r2,
        X_test_s, y_test
    )


if __name__ == "__main__":
    main()
