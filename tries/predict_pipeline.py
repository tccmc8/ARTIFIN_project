# RECOMMENDED IMPORTS

import mlflow
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np  
from typing import Optional
import sys
import kagglehub
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
scaler = joblib.load('scaler.joblib') 

# need to direct to mlruns for MLFLOW_TRACKING_URI and MLFLOW_REGISTRY_URI for this to work on terminal without using docker


# LOAD DATA

path = kagglehub.dataset_download("atharvasoundankar/futuristic-smart-city-citizen-activity-dataset")

print("Path to dataset files:", path)

DATA_DIR = Path("..") / "data"
file_path = DATA_DIR / "smart_city_citizen_activity.csv"

from kagglehub import KaggleDatasetAdapter

file_path = "smart_city_citizen_activity.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "atharvasoundankar/futuristic-smart-city-citizen-activity-dataset",
  file_path,)


# 