from pathlib import Path

import joblib
import pandas as pd

from src.train import build_pipeline
from sklearn.linear_model import LinearRegression


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



def test_training_pipeline_can_be_created():
    pipeline = build_pipeline(LinearRegression())

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_training_pipeline_can_fit_and_predict():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("price", axis=1).head(20)
    y = df["price"].head(20)

    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)

    assert len(predictions) == 20
    assert all(predictions > 0)
