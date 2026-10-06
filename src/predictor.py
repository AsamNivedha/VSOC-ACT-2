from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"


def load_model(path=MODEL_PATH):
    """Load the trained house price prediction pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


def predict_price(model, input_data):
    """Predict the house price for the supplied property data."""
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError("input_data must be a pandas DataFrame")

    if input_data.empty:
        raise ValueError("input_data must contain at least one row")

    predictions = model.predict(input_data)

    if len(predictions) == 0:
        raise ValueError("Model returned no predictions")

    return predictions
