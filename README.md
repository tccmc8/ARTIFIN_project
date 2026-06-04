# Student Habits vs Academic Performance — MLOps Pipeline

Predicts a student's **exam score** from daily habits (study hours, sleep, social media use, etc.) using two regression models tracked with **MLflow** and served via a **FastAPI** web service. Two models are compared: **Ridge Regression** and **Random Forest**. The best-performing model is saved as "best_model".

**Dataset:** [Student Habits vs Academic Performance](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance/data) (Kaggle)

---

## Project Structure

```
student-habits-mlops/
│
├── data/
│   └── raw/
│       └── student_habits_performance.csv
│   └── current_batches/            # Simulated incoming data batches
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── main.py
│   └── test_api.py
│
├── models/            # Generated after training
│   ├── best_model.joblib
│   ├── feature_columns.joblib
│   ├── best_model_info.json
│   └── category_maps.json
│
├── deploy/
│   ├── webservices/
│   │   └── Dockerfile         
│   └── batch/
│       ├── deploy.py   
│       └── schedule.py    
│  
├── monitoring/
│   ├── docker-compose.yaml       
│   └── scripts/
│       ├── prepare_reference.py    
│       ├── generate_batch.py       
│       └── calculate_metrics.py
│
├── mlruns/            # MLflow experiment tracking data
│
├── streaming/
│   ├── function/
│       ├── main.py
│       └── requirements.txt
│   └── ui/
│       ├── Dockerfile      # Cloud Run image for the streaming UI
│       ├── main.py
│       ├── requirements.txt
│       └── templates
│          └── index.html
|
├── EDA.ipynb          # Exploratory Data Analysis
├── cloudbuild.yaml         # Cloud Build config (builds app/Dockerfile)
├── requirements.txt
└── .gitignore
```
---

## Requirements

- Python 3.10+
- All dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Train the Models

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Tune **five base models** with 5-fold `GridSearchCV` under the `student-habits-performance` MLflow experiment:
   - **Ridge Regression** (linear baseline)
   - **Random Forest Regressor** (ensemble)a
  - **Lasso Regression** (L1-regularised linear with automatic feature selection)
  - **K-Nearest Neighbours** (instance-based non-parametric baseline)
  - **Random Forest Regressor** (bagged decision trees)
  - **XGBoost Regressor** (gradient-boosted trees)
- Add a **VotingRegressor** (averaging ensemble of the five tuned models), evaluated as a sixth run
- Log the parameter grid, chosen best hyperparameters, CV best RMSE, and test-set RMSE/MAE/R² for every run
- Save the best model (lowest test RMSE) to the `models/` folder
- Save `scaler.joblib` and `best_model.joblib` locally for the API

### 2. View MLflow Experiment Results

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser to compare runs.

In the browser you will see:
- The `Student_Habits_Academic_Performance` experiment with **two runs** (one per model).
- Metrics: `cv_rmse_mean`, `val_rmse`, `val_r2`, `test_rmse`, `test_r2`, etc.
- The registered model `BestStudentHabitsModel` under the **Models** tab.

### 3. Test a Prediction (Command Line)

```bash
python src/predict.py
```

Loads the saved model and prints an actual vs. predicted exam score for a sample student in the terminal or command line.

### 4. Start the API

```bash
uvicorn app.main:app --reload
```
#### Example POST request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 20,
    "gender": "Male",
    "study_hours_per_day": 6,
    "social_media_hours": 2,
    "netflix_hours": 1,
    "part_time_job": "No",
    "attendance_percentage": 90,
    "sleep_hours": 7,
    "diet_quality": "Good",
    "exercise_frequency": 3,
    "parental_education_level": "Bachelor",
    "internet_quality": "Good",
    "mental_health_rating": 8,
    "extracurricular_participation": "Yes"
  }'
```


---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns model name and feature count |
| `GET` | `/predict-sample/{index}` | Predict using a row from the dataset by index |
| `POST` | `/predict` | Predict from a manually provided JSON body |

---

## Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Student age |
| `gender` | Categorical | Male / Female / Other |
| `study_hours_per_day` | Numeric | Daily study hours |
| `social_media_hours` | Numeric | Daily social media usage |
| `netflix_hours` | Numeric | Daily streaming hours |
| `part_time_job` | Categorical | Yes / No |
| `attendance_percentage` | Numeric | % of classes attended |
| `sleep_hours` | Numeric | Hours of sleep per night |
| `diet_quality` | Categorical | Poor / Fair / Good |
| `exercise_frequency` | Numeric | Exercise sessions per week |
| `parental_education_level` | Categorical | High School / Bachelor / Master |
| `internet_quality` | Categorical | Poor / Average / Good |
| `mental_health_rating` | Numeric | Self-rated 0–10 |
| `extracurricular_participation` | Categorical | Yes / No |

Categorical features are encoded to integers before training. The mapping is saved in `models/category_maps.json`.

---

## Categorical Field Encoding

The API expects integer-encoded categorical values (matching the training encoding):

| Field                          | Encoding                                     |
|--------------------------------|----------------------------------------------|
| `gender`                       | 0 = Male, 1 = Female, 2 = Other              |
| `diet_quality`                 | 0 = Poor, 1 = Fair, 2 = Good                 |
| `parental_education_level`     | 0 = High School, 1 = Bachelor, 2 = Master    |
| `internet_quality`             | 0 = Poor, 1 = Average, 2 = Good              |
| `extracurricular_participation`| 0 = No, 1 = Yes                              |
| `part_time_job`                | 0 = No, 1 = Yes                              |

---

## Technologies

- **scikit-learn** — Machine learning models and preprocessing pipelines
- **MLflow** — Experiment tracking, parameter/metric logging, model registry
- **FastAPI** — REST API framework
- **Pydantic** — Request validation
- **pandas** — Data loading and manipulation
- **joblib** — Model serialisation
- **uvicorn** — ASGI web server to run FastAPI

---

## Reproducibility

All steps are reproducible from the raw dataset:
1. Clone this repository
2. Place `student_habits_performance.csv` in `data/raw/`
3. Run `python src/train.py` to train and save the model
4. Run the monitoring setup steps above to spin up the full observability stack

_Last demo run: Wed May 27 20:41:09 CEST 2026_

_Last demo run: Wed May 27 20:51:49 CEST 2026_
