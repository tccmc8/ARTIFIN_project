"""
Streaming UI for the Student Habits exam-score model.

Production deployment uses HTTP to talk to the prediction API. For the live
demo, HTTP keeps the moving parts simple. The Pub/Sub-based streaming
architecture is preserved in git history.
"""

import os
import uuid

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Student Habits Streaming UI")
templates = Jinja2Templates(directory="templates")

PREDICTION_API_URL = os.environ.get(
    "PREDICTION_API_URL",
    "https://student-habits-api-755586938880.us-central1.run.app/predict",
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    study_hours_per_day: float = Form(...),
    social_media_hours: float = Form(...),
    netflix_hours: float = Form(...),
    part_time_job: str = Form(...),
    attendance_percentage: float = Form(...),
    sleep_hours: float = Form(...),
    diet_quality: str = Form(...),
    exercise_frequency: int = Form(...),
    parental_education_level: str = Form(...),
    internet_quality: str = Form(...),
    mental_health_rating: int = Form(...),
    extracurricular_participation: str = Form(...),
):
    student_id = str(uuid.uuid4())[:8]

    features = {
        "age": age,
        "gender": gender,
        "study_hours_per_day": study_hours_per_day,
        "social_media_hours": social_media_hours,
        "netflix_hours": netflix_hours,
        "part_time_job": part_time_job,
        "attendance_percentage": attendance_percentage,
        "sleep_hours": sleep_hours,
        "diet_quality": diet_quality,
        "exercise_frequency": exercise_frequency,
        "parental_education_level": parental_education_level,
        "internet_quality": internet_quality,
        "mental_health_rating": mental_health_rating,
        "extracurricular_participation": extracurricular_participation,
    }

    prediction = await call_prediction_api(features, student_id)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": prediction,
            "submitted": features,
        },
    )


async def call_prediction_api(features: dict, student_id: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(PREDICTION_API_URL, json=features)
            response.raise_for_status()
            api_result = response.json()
            return {
                "prediction": {
                    "student_id": student_id,
                    "predicted_exam_score": api_result.get("predicted_exam_score"),
                }
            }
    except httpx.HTTPStatusError as exc:
        return {
            "prediction": {
                "student_id": student_id,
                "error": f"API returned {exc.response.status_code}: {exc.response.text}",
            }
        }
    except httpx.RequestError as exc:
        return {
            "error": f"Could not reach prediction API at {PREDICTION_API_URL}. ({exc})"
        }
