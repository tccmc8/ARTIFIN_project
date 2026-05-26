"""
Unit tests for prediction_core.py.

Run locally with:
    pytest test_prediction_core.py -v

These tests use dummy models so the CI pipeline does not need to download
the trained joblib model or load MLflow — same pattern as the teacher's
Iris example, but adapted for the 14-feature student-habits regression
problem.
"""

import sys
from pathlib import Path

# Make prediction_core importable when this file is run by pytest from the
# repository root (the way the CI workflow runs it).
sys.path.append(str(Path(__file__).parent))

import pytest

from prediction_core import (
    EXPECTED_FEATURE_COUNT,
    clip_score,
    predict,
    validate_input,
)


# ──────────────────────────────────────────
# Dummy models — stand in for the real sklearn model in CI
# ──────────────────────────────────────────

class DummyModel:
    """Always returns a fixed exam-score prediction, regardless of input."""

    def predict(self, X):
        return [75.0]


class ShapeCheckingDummyModel:
    """Asserts the input arrives as a single row with 14 columns."""

    def predict(self, X):
        assert len(X) == 1
        assert len(X[0]) == EXPECTED_FEATURE_COUNT
        return [82.5]


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def make_valid_sample():
    """
    Build a valid 14-feature row in the same column order as
    feature_columns.joblib produced by train.py.

    Categorical fields are already integer-encoded using category_maps.json:
    gender=Male=0, part_time_job=No=0, diet_quality=Good=2, etc.
    """
    return [
        20,    # age
        0,     # gender (Male)
        6.0,   # study_hours_per_day
        2.0,   # social_media_hours
        1.0,   # netflix_hours
        0,     # part_time_job (No)
        90.0,  # attendance_percentage
        7.0,   # sleep_hours
        2,     # diet_quality (Good)
        3,     # exercise_frequency
        1,     # parental_education_level (Bachelor)
        2,     # internet_quality (Good)
        8,     # mental_health_rating
        1,     # extracurricular_participation (Yes)
    ]


# ──────────────────────────────────────────
# Tests for predict() + validate_input()
# ──────────────────────────────────────────

def test_valid_prediction():
    sample = make_valid_sample()
    model = DummyModel()

    result = predict(sample, model)

    assert result == [75.0]


def test_invalid_feature_length_too_few():
    sample = [20, 0, 6.0]  # only 3 features

    with pytest.raises(ValueError):
        validate_input(sample)


def test_invalid_feature_length_too_many():
    sample = make_valid_sample() + [99]  # 15 features

    with pytest.raises(ValueError):
        validate_input(sample)


def test_non_numeric_input_rejects_strings():
    sample = make_valid_sample()
    sample[1] = "Male"  # raw text instead of encoded int

    with pytest.raises(ValueError):
        validate_input(sample)


def test_non_numeric_input_rejects_bool():
    # booleans must be encoded to 0/1 ints by category_maps before predict
    sample = make_valid_sample()
    sample[5] = True

    with pytest.raises(ValueError):
        validate_input(sample)


def test_non_numeric_input_rejects_none():
    sample = make_valid_sample()
    sample[3] = None

    with pytest.raises(ValueError):
        validate_input(sample)


def test_model_input_shape():
    """The predict() helper must hand the model a 2D batch of size 1×14."""
    sample = make_valid_sample()
    model = ShapeCheckingDummyModel()

    result = predict(sample, model)

    assert result == [82.5]


# ──────────────────────────────────────────
# Tests for clip_score()
# ──────────────────────────────────────────

def test_clip_score_within_range():
    assert clip_score(50.0) == 50.0


def test_clip_score_above_range_is_capped():
    assert clip_score(150.0) == 100.0


def test_clip_score_below_range_is_floored():
    assert clip_score(-10.0) == 0.0


def test_clip_score_accepts_ints():
    assert clip_score(75) == 75.0
