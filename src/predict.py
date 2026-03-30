from pathlib import Path
import json

import joblib
import pandas as pd


# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_habits_performance.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------- Constants ----------
TARGET_COLUMN = "exam_score"
SAMPLE_INDEX = 2


def load_model():
    """ Load the saved best model. """

    model_path = MODELS_DIR / "best_model.joblib"
    return joblib.load(model_path)


def load_feature_columns():
    """ Load the saved feature column names. """

    features_path = MODELS_DIR / "feature_columns.joblib"
    return joblib.load(features_path)


def load_category_maps():
    """ Load saved category mappings. """
    
    maps_path = MODELS_DIR / "category_maps.json"
    with open(maps_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_dataframe(df: pd.DataFrame, category_maps):
    """ Prepare the raw dataset in the same way as in train. """
    
    df = df.copy()

    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    if "parental_education_level" in df.columns:
        df["parental_education_level"] = df["parental_education_level"].fillna(
            df["parental_education_level"].mode()[0]
        )

    for column, mapping in category_maps.items():
        df[column] = df[column].astype(str).str.strip().str.title()
        df[column] = df[column].map(mapping)

    return df


def load_sample_data(feature_columns, category_maps, sample_index=2):
    """ The same row , 2 in this instance, is extracted from the dataset to make a
    prediction and compare it to the actual exam score. There is an if clause to check 
    that the sample row is within the dataset bounds. """
    
    df = pd.read_csv(DATA_PATH)
    df = prepare_dataframe(df, category_maps)

    if sample_index < 0 or sample_index >= len(df):
        raise IndexError(f"sample_index must be between 0 and {len(df) - 1}")

    X_sample = df.loc[[sample_index], feature_columns]
    y_actual = float(df.loc[sample_index, TARGET_COLUMN])

    return X_sample, y_actual


def predict_sample(model, X_sample):
    """ Does the actual prediction for one sample using the model. """
    
    predicted_score = float(model.predict(X_sample)[0])
    return predicted_score


def show_feature_weights(model, feature_columns):
    """ Show feature weights for the saved Ridge model. """
    
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["model"]

    weights_df = pd.DataFrame({
        "feature": feature_columns,
        "coef_scaled": ridge.coef_,
        "effect_per_original_unit": ridge.coef_ / scaler.scale_
    })

    weights_df["abs_effect"] = weights_df["effect_per_original_unit"].abs()
    weights_df = weights_df.sort_values("abs_effect", ascending=False)

    print("\nFeature weights")
    print("---------------")
    print(weights_df[["feature", "effect_per_original_unit"]])

    return weights_df


def main():
    print("Loading saved model...")
    model = load_model()

    print("Loading feature columns...")
    feature_columns = load_feature_columns()

    print("Showing feature weights...")
    show_feature_weights(model, feature_columns)

    print("Loading category maps...")
    category_maps = load_category_maps()

    print("Loading one sample from dataset...")
    X_sample, y_actual = load_sample_data(
        feature_columns,
        category_maps,
        sample_index=SAMPLE_INDEX
    )

    print("Making prediction...")
    predicted_score = predict_sample(model, X_sample)

    print("\nPrediction result")
    print("-----------------")
    print("Sample index:", SAMPLE_INDEX)
    print("Actual exam score:", round(y_actual, 2))
    print("Predicted exam score:", round(predicted_score, 2))


if __name__ == "__main__":
    main()