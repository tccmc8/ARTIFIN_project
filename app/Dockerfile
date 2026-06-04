# Dockerfile for Cloud Run deployment of the Student Habits API.
# Differs from the main Dockerfile in that it bakes the trained models
# directly into the image, so the container is self-contained and doesn't
# need GCS or volume mounts at runtime.

FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first so this layer is cached when only
# code or models change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app and the trained models. The data CSV is also
# needed because /predict-sample reads it at request time.
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Cloud Run injects the port to listen on via the PORT env var
# (defaults to 8080). The app reads $PORT; default 8080 if unset.
ENV PORT=8080
EXPOSE 8080

# Use shell form so $PORT gets expanded at container start.
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
