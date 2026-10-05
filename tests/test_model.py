from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "house_price_pipeline.pkl"
DATA_PATH = BASE_DIR / "data.csv"


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame({
        "area": [5000],
        "bedrooms": [3],
        "bathrooms": [2],
        "stories": [2],
        "mainroad": ["yes"],
        "guestroom": ["no"],
        "basement": ["no"],
        "hotwaterheating": ["no"],
        "airconditioning": ["yes"],
        "parking": [2],
        "prefarea": ["yes"],
        "furnishingstatus": ["semi-furnished"],
    })

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] > 0


def test_dataset_has_required_columns():
    df = pd.read_csv(DATA_PATH)

    required_columns = {
        "price",
        "area",
        "bedrooms",
        "bathrooms",
        "stories",
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "parking",
        "prefarea",
        "furnishingstatus",
    }

    assert required_columns.issubset(df.columns)

