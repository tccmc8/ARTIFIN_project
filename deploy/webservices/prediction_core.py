"""
Core prediction logic for the Student Habits exam-score model.

This module is intentionally tiny and free of heavy ML dependencies so that
the CI pipeline can import it and run the unit tests in a clean Ubuntu
environment without needing the full sklearn/MLflow stack.

The same pattern is used in the teacher's Iris example
(03_dockerization_and_deployment/webservices/prediction_core.py), adapted
here for a 14-feature regression problem instead of 4-feature classification.
"""

# Number of features expected after categorical encoding.
# Order must match feature_columns.joblib saved during training.
EXPECTED_FEATURE_COUNT = 14

# Realistic bounds for the predicted exam score (the target is 0–100).
SCORE_MIN = 0.0
SCORE_MAX = 100.0


def validate_input(features):
    """
    Validate a single feature vector before it is passed to the model.

    Checks:
    - the vector has exactly EXPECTED_FEATURE_COUNT entries
    - every entry is numeric (int or float)

    Booleans are rejected on purpose: categorical fields like ``gender`` or
    ``part_time_job`` must be integer-encoded (using category_maps.json)
    before they reach the model, the same way training does it.
    """
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Student-habits input must contain exactly "
            f"{EXPECTED_FEATURE_COUNT} features, got {len(features)}."
        )

    for value in features:
        # ``bool`` is a subclass of ``int`` in Python, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("All features must be numeric (int or float).")


def clip_score(score):
    """Clip a predicted exam score to the valid 0–100 range."""
    return max(SCORE_MIN, min(SCORE_MAX, float(score)))


def predict(features, model):
    """
    Validate the features and return the model's prediction.

    The model parameter is anything with a ``.predict`` method that accepts
    a 2D array-like, matching the scikit-learn estimator interface used by
    Ridge and RandomForestRegressor in train.py.
    """
    validate_input(features)
    return model.predict([features])
