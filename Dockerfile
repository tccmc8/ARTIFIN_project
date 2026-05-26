# Dockerfile for the Student Habits FastAPI prediction service.
# Build context: repo root (so `docker build .` finds requirements.txt and app/).
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first so this layer is cached when only
# app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app package.
COPY app/ ./app/

# Document the port the app listens on.
EXPOSE 8000

# At runtime the container expects a `models/` directory (with the
# trained joblib files and JSON config) to be mounted at /app/models.
# Without it the FastAPI startup will fail because main.py loads those
# at import time.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
