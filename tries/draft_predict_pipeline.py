# this code is not mine, it is from a lecture. I am annotating it to get an understanding of what I need to do.


# recommend imports

import mlflow
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np  
from typing import Optional
import sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
scaler = joblib.load('scaler.joblib') 

# need to direct to mlruns for MLFLOW_TRACKING_URI and MLFLOW_REGISTRY_URI for this to work on terminal without using docker


loaded_model = mlflow.pyfunc.load_model(MODEL_DIR.as_posix())

# MODEL_DIR is the ML runs folder

def predict(input_data):
    # Always convert first
    input_data = np.array(input_data)
    # If user passes a single sample like [5.1, 3.5, 1.4, 0.2]
    # turn it into shape (1, 4)
    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    # apply the scaler to the input data
    input_data = scaler.transform(input_data)
    return loaded_model.predict(input_data)

# need to look into this part in more depth an detail

app = FastAPI()

class PredictRequest(BaseModel):
    run_id: Optional[str] = None
    input_data: list[float]

# this basically gives you the structure or what you want as you input from the user/ client. In this case it is a list of float (0.00 - decimal) numbers. I think that you could set a limit or even do some UI and include boxes of what you want and where. In the case of my dataset it could be a float number for the home energy usage and an integer for the age and hours spent doing something. The tricky part would be gender and vehicle - maybe tick boxes so integrate a user interface to get data.

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    input_array = np.array(req.input_data).reshape(1, -1)
    prediction = predict(input_array)
    return {"prediction": prediction.tolist()}

# this is basically asking for the prediction and will not let the prediction go through unless the requirements are met.


# To run the FastAPI run "uvicorn predict:app --reload" in your terminal 
