"""
Streaming UI for the Student Habits exam-score model.

Lets a user fill in 14 habit-related fields, publishes the request to a
Pub/Sub topic, and waits (up to a few seconds) for the corresponding
prediction to come back on a separate subscription.

Mirrors the structure of the teacher's Iris UI
(04_streaming/ui/main.py) — same publish/pull-loop pattern, adapted for
the 14 fields and the regression output.
"""

import json
import os
import time
import uuid

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from google.cloud import pubsub_v1


app = FastAPI(title="Student Habits Streaming UI")
templates = Jinja2Templates(directory="templates")


# ──────────────────────────────────────────
# Environment variables (set when deploying the container)
# ──────────────────────────────────────────
PROJECT_ID = os.environ["PROJECT_ID"]
INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "student-habits-features")
PREDICTION_SUBSCRIPTION = os.environ.get(
    "PREDICTION_SUBSCRIPTION",
    "student-habits-ui-predictions-sub",
)


publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()

topic_path = publisher.topic_path(PROJECT_ID, INPUT_TOPIC)
subscription_path = subscriber.subscription_path(
    PROJECT_ID,
    PREDICTION_SUBSCRIPTION,
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
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
    student_id = str(uuid.uuid4())

    message = {
        "student_id": student_id,
        "features": {
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
        },
    }

    publisher.publish(
        topic_path,
        json.dumps(message).encode("utf-8"),
    )

    prediction = wait_for_prediction(student_id)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": prediction,
            "submitted": message["features"],
        },
    )


def wait_for_prediction(student_id: str, timeout: int = 10):
    """
    Poll the prediction subscription for a message whose student_id matches.
    Acknowledge every message we pull (matching or not) so the queue does
    not back up between calls.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        response = subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": 10,
            }
        )

        ack_ids = []
        match = None

        for msg in response.received_messages:
            payload = json.loads(msg.message.data.decode("utf-8"))
            pred = payload.get("prediction", {})
            ack_ids.append(msg.ack_id)

            if pred.get("student_id") == student_id:
                match = payload

        if ack_ids:
            subscriber.acknowledge(
                request={
                    "subscription": subscription_path,
                    "ack_ids": ack_ids,
                }
            )

        if match is not None:
            return match

        time.sleep(1)

    return {"error": "Prediction timeout"}
