# train_pipeline.py
# Mirrors the StudentHabits_MLflow_Experiments.ipynb exactly.
#
# Two separate MLflow experiments:
#   1. Student_Habits_Ridge_Regression  — sweeps alphas [0.1, 1.0, 10.0, 100.0]
#   2. Student_Habits_Random_Forest     — sweeps n_estimators [50, 100, 200, 300]
#
# Uses a SQLite backend (mlflow.db) to avoid the file-store deprecation and the
# 'no artifacts' bug present in newer MLflow versions.
#
# Usage:
#   cd /path/to/your/project
#   python train_pipeline.py
#
# Then start the UI with:
#   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ── Imports ───────────────────────────────────────────────────────────────────
import pathlib
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

# ── Absolute paths — anchored to this file's directory ───────────────────────
PROJECT_DIR  = pathlib.Path(__file__).resolve().parent
DB_PATH      = PROJECT_DIR / "mlflow.db"
SCALER_PATH  = PROJECT_DIR / "scaler.joblib"
MODEL_PATH   = PROJECT_DIR / "best_model.joblib"

# SQLite URI — no file-store deprecation warnings, no missing-artifact bug
MLFLOW_TRACKING_URI = f"sqlite:///{DB_PATH}"


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
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
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")
    df.info()


# ── 3. Preprocessing ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["parental_education_level"] = df["parental_education_level"].fillna(
        df["parental_education_level"].mode()[0]
    )
    print("Missing values after fill:")
    print(df.isnull().sum())

    df["gender"]                       = df["gender"].str.strip().str.title().map({"Male": 0, "Female": 1, "Other": 2})
    df["diet_quality"]                  = df["diet_quality"].str.strip().str.title().map({"Poor": 0, "Fair": 1, "Good": 2})
    df["parental_education_level"]      = df["parental_education_level"].str.strip().str.title().map({"High School": 0, "Bachelor": 1, "Master": 2})
    df["internet_quality"]              = df["internet_quality"].str.strip().str.title().map({"Poor": 0, "Average": 1, "Good": 2})
    df["extracurricular_participation"] = df["extracurricular_participation"].str.strip().str.title().map({"No": 0, "Yes": 1})
    df["part_time_job"]                 = df["part_time_job"].str.strip().str.title().map({"No": 0, "Yes": 1})

    df = df.drop(columns=["student_id"], errors="ignore")
    return df


# ── 4. Train / validation / test split ───────────────────────────────────────
def train_val_test_split(df: pd.DataFrame):
    X = df.drop(columns=["exam_score"])
    y = df["exam_score"]
    y_binned = pd.cut(y, bins=5, labels=False)

    X_train, X_temp, y_train, y_temp, yb_train, yb_temp = train_test_split(
        X, y, y_binned, test_size=0.40, random_state=42, stratify=y_binned
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=yb_temp
    )
    print(f"Train set:      {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 5. Scale features ─────────────────────────────────────────────────────────
def fit_and_scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


# ── 6. MLflow setup ───────────────────────────────────────────────────────────
def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())
    print("Database file      :", DB_PATH)
    print()
    print("To open the UI run (from any directory):")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5000")


# ── 7. Experiment 1 — Ridge Regression ───────────────────────────────────────
def run_ridge_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test):
    mlflow.set_experiment("Student_Habits_Ridge_Regression")

    for alpha in [0.1, 1.0, 10.0, 100.0]:
        with mlflow.start_run(run_name=f"Ridge_alpha_{alpha}"):
            model = Ridge(alpha=alpha, random_state=42)
            model.fit(X_train_s, y_train)

            y_val_pred  = model.predict(X_val_s)
            y_test_pred = model.predict(X_test_s)

            val_rmse  = float(np.sqrt(mean_squared_error(y_val,  y_val_pred)))
            val_mae   = float(mean_absolute_error(y_val,  y_val_pred))
            val_r2    = float(r2_score(y_val,  y_val_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            test_mae  = float(mean_absolute_error(y_test, y_test_pred))
            test_r2   = float(r2_score(y_test, y_test_pred))

            mlflow.log_param("model_type",   "Ridge")
            mlflow.log_param("alpha",        alpha)
            mlflow.log_param("random_state", 42)

            mlflow.log_metric("val_rmse",  val_rmse)
            mlflow.log_metric("val_mae",   val_mae)
            mlflow.log_metric("val_r2",    val_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae",  test_mae)
            mlflow.log_metric("test_r2",   test_r2)

            # Use name= keyword — fixes the 'artifact_path deprecated' warning
            mlflow.sklearn.log_model(sk_model=model, name="model")

            run_id = mlflow.active_run().info.run_id
            print(f"alpha={alpha:<6}  val_r2={val_r2:.4f}  test_rmse={test_rmse:.4f}  run_id={run_id}")

        mlflow.register_model(f"runs:/{run_id}/model", f"student_ridge_alpha_{alpha}")

    ridge_runs = mlflow.search_runs(
        experiment_names=["Student_Habits_Ridge_Regression"],
        order_by=["metrics.test_rmse ASC"]
    )
    best             = ridge_runs.iloc[0]
    best_run_id      = best["run_id"]
    best_alpha       = best["params.alpha"]
    best_test_rmse   = best["metrics.test_rmse"]
    best_test_r2     = best["metrics.test_r2"]

    print(f"\nBest Ridge:  alpha={best_alpha}  test_rmse={best_test_rmse:.4f}  test_r2={best_test_r2:.4f}")
    mlflow.register_model(f"runs:/{best_run_id}/model", "best_student_ridge")
    print("Best Ridge model registered as 'best_student_ridge'")
    return best_run_id, best_alpha, best_test_rmse, best_test_r2


# ── 8. Experiment 2 — Random Forest ──────────────────────────────────────────
def run_rf_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test):
    mlflow.set_experiment("Student_Habits_Random_Forest")

    for n_est in [50, 100, 200, 300]:
        with mlflow.start_run(run_name=f"RF_n_estimators_{n_est}"):
            model = RandomForestRegressor(
                n_estimators=n_est, max_depth=10, min_samples_leaf=4,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train_s, y_train)

            y_val_pred  = model.predict(X_val_s)
            y_test_pred = model.predict(X_test_s)

            val_rmse  = float(np.sqrt(mean_squared_error(y_val,  y_val_pred)))
            val_mae   = float(mean_absolute_error(y_val,  y_val_pred))
            val_r2    = float(r2_score(y_val,  y_val_pred))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            test_mae  = float(mean_absolute_error(y_test, y_test_pred))
            test_r2   = float(r2_score(y_test, y_test_pred))

            mlflow.log_param("model_type",       "RandomForest")
            mlflow.log_param("n_estimators",     n_est)
            mlflow.log_param("max_depth",        10)
            mlflow.log_param("min_samples_leaf", 4)
            mlflow.log_param("random_state",     42)

            mlflow.log_metric("val_rmse",  val_rmse)
            mlflow.log_metric("val_mae",   val_mae)
            mlflow.log_metric("val_r2",    val_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_mae",  test_mae)
            mlflow.log_metric("test_r2",   test_r2)

            # Use name= keyword — fixes the 'artifact_path deprecated' warning
            mlflow.sklearn.log_model(sk_model=model, name="model")

            run_id = mlflow.active_run().info.run_id
            print(f"n_estimators={n_est:<4}  val_r2={val_r2:.4f}  test_rmse={test_rmse:.4f}  run_id={run_id}")

        mlflow.register_model(f"runs:/{run_id}/model", f"student_rf_{n_est}_trees")

    rf_runs = mlflow.search_runs(
        experiment_names=["Student_Habits_Random_Forest"],
        order_by=["metrics.test_rmse ASC"]
    )
    best           = rf_runs.iloc[0]
    best_run_id    = best["run_id"]
    best_n_est     = best["params.n_estimators"]
    best_test_rmse = best["metrics.test_rmse"]
    best_test_r2   = best["metrics.test_r2"]

    print(f"\nBest RF:  n_estimators={best_n_est}  test_rmse={best_test_rmse:.4f}  test_r2={best_test_r2:.4f}")
    mlflow.register_model(f"runs:/{best_run_id}/model", "best_student_rf")
    print("Best Random Forest model registered as 'best_student_rf'")
    return best_run_id, best_n_est, best_test_rmse, best_test_r2


# ── 9. Compare, register overall best, alias to Staging ──────────────────────
def register_best_model(
    best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2,
    best_rf_run_id,    best_rf_n_est,    best_rf_test_rmse,    best_rf_test_r2,
    X_test_s, y_test
):
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

    mlflow.register_model(f"runs:/{overall_best_run_id}/model", "BestStudentHabitsModel")
    print("Overall best model registered as 'BestStudentHabitsModel'")

    # Use alias instead of deprecated transition_model_version_stage
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    date   = datetime.today().strftime("%Y-%m-%d")

    client.set_registered_model_alias(
        name="BestStudentHabitsModel",
        alias="Staging",
        version=1
    )
    print("Model 'BestStudentHabitsModel' version 1 aliased as 'Staging'")

    client.update_model_version(
        name="BestStudentHabitsModel",
        version=1,
        description=(
            f"Best model from Student Habits experiments: {overall_best_label}. "
            f"test_rmse={min(best_ridge_test_rmse, best_rf_test_rmse):.4f} as of {date}"
        )
    )
    print("Model description updated.")

    # Evaluate using the alias URI
    loaded_model = mlflow.pyfunc.load_model("models:/BestStudentHabitsModel@Staging")
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

    # Save sklearn model locally for FastAPI
    best_sklearn_model = mlflow.sklearn.load_model(f"runs:/{overall_best_run_id}/model")
    joblib.dump(best_sklearn_model, MODEL_PATH)
    print(f"\nbest_model.joblib saved to {MODEL_PATH}")
    print("Ready for predict_pipeline.py")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    check_missing_values(df)
    df = preprocess(df)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)
    X_train_s, X_val_s, X_test_s = fit_and_scale(X_train, X_val, X_test)

    setup_mlflow()

    best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2 = \
        run_ridge_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    best_rf_run_id, best_rf_n_est, best_rf_test_rmse, best_rf_test_r2 = \
        run_rf_experiment(X_train_s, X_val_s, X_test_s, y_train, y_val, y_test)

    register_best_model(
        best_ridge_run_id, best_ridge_alpha, best_ridge_test_rmse, best_ridge_test_r2,
        best_rf_run_id,    best_rf_n_est,    best_rf_test_rmse,    best_rf_test_r2,
        X_test_s, y_test
    )


if __name__ == "__main__":
    main()
