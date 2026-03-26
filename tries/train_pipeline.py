# RECOMMENDED IMPORTS

import mlflow
import mlflow.sklearn
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np  
from typing import Optional
import sys
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt`
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# LOAD DATA
def load_data():
    """ load the dataset from the downloaded kaggle csv file """
    
    path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")
    print("Path to dataset files:", path)
    DATA_DIR = Path("..") / "data"
    file_path = DATA_DIR / "student_habits_performance.csv"
    file_path = "student_habits_performance.csv"
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "jayaantanaath/student-habits-vs-academic-performance",
        file_path,)
    
    return df


# PREPARE DATA
def check_missing_values(df):
    """ check the data for missing values and show the missing values, returning
the missing values """

    missing_val = df.isnull().sum()
    print("missing values in each column:")
    print(missing_val)
	
    return missing_val

def fill_missing_education(df):
    """ In the dataset there are a great number of missing terms in the education field
for parents, we are working under the assumption that the parents would have a high school
eduction. This function fills the null values with highschool as the level of education """

    df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])
    
    return df['parental_education_level']

def intergerise(df):
    """ Turning categories like male, female and other or poor, average, good etc. into
integers so they are easier to work with, using a dictionary to assign the internet quality,
extracuricular participation, parental education, diet wuality and gender a number """

    df["gender"].unique()
    df["gender"] = df["gender"].str.strip().str.title()
    gender_map = {
        "Male": 0,
        "Female": 1,
        "Other": 2
        }
    df["gender"] = df["gender"].map(gender_map)

    df["diet_quality"].unique()
    df["diet_quality"] = df["diet_quality"].str.strip().str.title()
    diet_map = {
        "Poor": 0,
        "Fair": 1,
        "Good": 2
        }
    df["diet_quality"] = df["diet_quality"].map(diet_map)

    df["parental_education_level"].unique()
    df["parental_education_level"] = df["parental_education_level"].str.strip().str.title()
    par_ed_lev_map = {
        "High School": 0,
        "Bachelor": 1,
        "Master": 2
        }
    df["parental_education_level"] = df["parental_education_level"].map(par_ed_lev_map)

    df["internet_quality"].unique()
    df["internet_quality"] = df["internet_quality"].str.strip().str.title()
    internet_map = {
        "Poor": 0,
        "Average": 1,
        "Good": 2
        }
    df["internet_quality"] = df["internet_quality"].map(internet_map)

    df["extracurricular_participation"].unique()
    df["extracurricular_participation"] = df["extracurricular_participation"].str.strip().str.title()
    clubs_map = {
        "No": 0,
        "Yes": 1
        }
    df["extracurricular_participation"] = df["extracurricular_participation"].map(clubs_map)

    return df["gender"], df["diet_quality"], df["parental_education_level"], df["internet_quality"], df["extracurricular_participation"]


# TRAIN AND TEST SPLIT
Train test split / train your model

def train_models(X_train, X_val, X_test, y_train, y_val, y_test):
	model directory
	use mlflows to track the experiments and store it in mlruns
	name/ set up the experiment
	start prediction and log the parameters
	get the accuracy of each predication and model

def train_test_val_split(df):

    TARGET = "exam_score"
    DROP   = ["student_id", TARGET]
    X = df.drop(columns=DROP)
    y = df[TARGET]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


    return X_train, X_val, X_test, y_train, y_val, y_test

def standardizer (X_train):
    # Fit scaler on training data (do this once after training)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler 

def train_and_log_models(X_train, X_val, X_test, y_train, y_val, y_test):

EXPERIMENT_NAME = "Study habits and academic results"
MLRUNS_DIR      = "./mlruns"
CV_FOLDS        = 5

os.makedirs(MLRUNS_DIR, exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

kfold   = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
results = {}   # model_key -> {label, rmse, mae, r2}


def eval_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae":  mean_absolute_error(y_true, y_pred),
        "r2":   r2_score(y_true, y_pred),
    }


def log_run(model, model_type_label, extra_params=None):
    
    """ K-fold CV on X_train, fit on full X_train, evaluate on
    X_val + X_test, log everything to the active MLflow run.
    Returns (cv_rmse_array, test_metrics_dict, fitted_model)"""
    
    cv_scores = -cross_val_score(
        model, X_train, y_train,
        cv=kfold, scoring="neg_root_mean_squared_error"
    )  # returns positive RMSE values

    model.fit(X_train, y_train)
    val_m  = eval_metrics(y_val,  model.predict(X_val))
    test_m = eval_metrics(y_test, model.predict(X_test))

    mlflow.log_param("model_type",   model_type_label)
    mlflow.log_param("cv_folds",     CV_FOLDS)
    mlflow.log_param("random_state", 42)
    if extra_params:
        for k, v in extra_params.items():
            mlflow.log_param(k, v)

    mlflow.log_metric("cv_rmse_mean", cv_scores.mean())
    mlflow.log_metric("cv_rmse_std",  cv_scores.std())
    for split, m in [("val", val_m), ("test", test_m)]:
        mlflow.log_metric(f"{split}_rmse", m["rmse"])
        mlflow.log_metric(f"{split}_mae",  m["mae"])
        mlflow.log_metric(f"{split}_r2",   m["r2"])

    return cv_scores, test_m, model
