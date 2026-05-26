"""
Integration tests for the FastAPI prediction service.

Place this file next to main.py (e.g. inside app/).

These tests use FastAPI's TestClient to exercise the real /predict and /
endpoints without binding to a port. To avoid depending on a trained
joblib model being committed to the repository — or having to train one
during CI — a session-scoped fixture writes a small set of dummy model
artefacts to MODELS_DIR before main.py is imported.

Any real artefacts that already exist are backed up before the test runs
and restored afterwards, so running this test locally never destroys a
model you've trained.

Run locally with:
    pytest app/test_api.py -v
"""

import json
import sys
from pathlib import Path

import joblib
import pytest


# Compute MODELS_DIR the same way main.py does. main.py uses
# parent.parent because it expects to live inside app/, so this file
# must be in the same directory for the paths to line up.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ──────────────────────────────────────────
# Dummy model — replaces the real trained pipeline so the test does not
# need scikit-learn to do any work. Must be defined at module scope (not
# inside a function) so joblib can pickle and unpickle it cleanly.
# ──────────────────────────────────────────

class DummyModel:
    """Returns a constant exam-score prediction regardless of the input."""

    def predict(self, X):
        return [75.0 for _ in X]


# ──────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def fake_model_artifacts():
    """
    Write dummy model artefacts into MODELS_DIR before main.py is imported,
    backing up any existing files first.

    autouse=True means this runs before any test in the file, including
    before the `client` fixture imports the FastAPI app.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    artefact_names = [
        "best_model.joblib",
        "feature_columns.joblib",
        "best_model_info.json",
        "category_maps.json",
    ]

    # Back up anything already there.
    backup = {}
    for name in artefact_names:
        p = MODELS_DIR / name
        if p.exists():
            backup[name] = p.read_bytes()

    # Build the fake artefacts.
    feature_columns = [
        "age", "gender", "study_hours_per_day", "social_media_hours",
        "netflix_hours", "part_time_job", "attendance_percentage",
        "sleep_hours", "diet_quality", "exercise_frequency",
        "parental_education_level", "internet_quality",
        "mental_health_rating", "extracurricular_participation",
    ]
    category_maps = {
        "gender": {"Male": 0, "Female": 1, "Other": 2},
        "part_time_job": {"No": 0, "Yes": 1},
        "diet_quality": {"Poor": 0, "Fair": 1, "Good": 2},
        "parental_education_level": {"High School": 0, "Bachelor": 1, "Master": 2},
        "internet_quality": {"Poor": 0, "Average": 1, "Good": 2},
        "extracurricular_participation": {"No": 0, "Yes": 1},
    }
    model_info = {
        "best_model_name": "DummyModel",
        "selection_metric": "rmse",
        "metrics": {"rmse": 0.0, "mae": 0.0, "r2": 1.0},
    }

    joblib.dump(DummyModel(), MODELS_DIR / "best_model.joblib")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.joblib")
    (MODELS_DIR / "best_model_info.json").write_text(json.dumps(model_info))
    (MODELS_DIR / "category_maps.json").write_text(json.dumps(category_maps))

    yield

    # Restore originals (or remove the dummies if no original existed).
    for name in artefact_names:
        p = MODELS_DIR / name
        if name in backup:
            p.write_bytes(backup[name])
        else:
            p.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def client():
    """Build the FastAPI TestClient after the dummy artefacts are in place."""
    sys.path.insert(0, str(HERE))
    from fastapi.testclient import TestClient
    from main import app  # imported after fake_model_artifacts has run
    return TestClient(app)


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def make_valid_payload():
    """A minimal valid request body matching PredictionRequest."""
    return {
        "age": 20,
        "gender": "Male",
        "study_hours_per_day": 6.0,
        "social_media_hours": 2.0,
        "netflix_hours": 1.0,
        "part_time_job": "No",
        "attendance_percentage": 90.0,
        "sleep_hours": 7.0,
        "diet_quality": "Good",
        "exercise_frequency": 3,
        "parental_education_level": "Bachelor",
        "internet_quality": "Good",
        "mental_health_rating": 8,
        "extracurricular_participation": "Yes",
    }


# ──────────────────────────────────────────
# Tests for GET /
# ──────────────────────────────────────────

def test_home_endpoint_responds(client):
    response = client.get("/")
    assert response.status_code == 200


def test_home_reports_correct_feature_count(client):
    response = client.get("/")
    body = response.json()
    assert body["number_of_features"] == 14


def test_home_returns_model_name(client):
    response = client.get("/")
    body = response.json()
    assert "best_model_name" in body
    assert isinstance(body["best_model_name"], str)


# ──────────────────────────────────────────
# Tests for POST /predict — happy path and output format
# ──────────────────────────────────────────

def test_predict_returns_200_for_valid_payload(client):
    response = client.post("/predict", json=make_valid_payload())
    assert response.status_code == 200


def test_predict_response_has_expected_keys(client):
    response = client.post("/predict", json=make_valid_payload())
    body = response.json()
    assert "predicted_exam_score" in body
    assert "best_model_name" in body


def test_predict_score_is_numeric(client):
    response = client.post("/predict", json=make_valid_payload())
    score = response.json()["predicted_exam_score"]
    assert isinstance(score, (int, float))


def test_predict_score_is_in_valid_range(client):
    """The endpoint clips to [0, 100] — verify that contract holds."""
    response = client.post("/predict", json=make_valid_payload())
    score = response.json()["predicted_exam_score"]
    assert 0.0 <= score <= 100.0


# ──────────────────────────────────────────
# Tests for POST /predict — error handling
# ──────────────────────────────────────────

def test_predict_rejects_missing_required_field(client):
    """Pydantic returns 422 when a required field is missing."""
    payload = make_valid_payload()
    del payload["age"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_out_of_range_numeric(client):
    """Field constraints (ge/le) catch numeric values outside their bounds."""
    payload = make_valid_payload()
    payload["mental_health_rating"] = 15  # max is 10
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_unknown_category(client):
    """An unrecognised categorical value is a 400, not a 500."""
    payload = make_valid_payload()
    payload["gender"] = "Robot"
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_accepts_lowercase_category(client):
    """Categorical fields are title-cased by the validator before lookup."""
    payload = make_valid_payload()
    payload["gender"] = "male"  # lowercase — should be normalised to "Male"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
