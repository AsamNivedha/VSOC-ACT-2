# House Price Prediction

A machine learning project for predicting house prices using multiple regression models, automated preprocessing, model evaluation, and a Streamlit web application.

The project is designed around a reusable scikit-learn pipeline so that the same preprocessing and trained model can be used during prediction.

---

## Overview

This project uses a housing dataset containing property characteristics such as:

- Area
- Number of bedrooms
- Number of bathrooms
- Number of stories
- Parking spaces
- Main road access
- Guest room availability
- Basement availability
- Hot water heating
- Air conditioning
- Preferred area
- Furnishing status

The target variable is the house **price**.

The training pipeline compares multiple regression algorithms using 5-fold cross-validation, selects the best-performing model based on cross-validation R², evaluates the selected model on a held-out test set, and saves the trained pipeline for use by the Streamlit application.

---

## Features

- Dataset validation before training
- Duplicate-row removal
- Automatic preprocessing of numerical and categorical features
- One-hot encoding for categorical features
- Multiple regression model comparison
- 5-fold cross-validation
- Automatic model selection
- Held-out test-set evaluation
- Reusable scikit-learn preprocessing and prediction pipeline
- Persisted trained model
- Persisted evaluation metrics in JSON format
- Streamlit web interface for predictions
- Unit and integration tests with pytest
- GitHub Actions continuous integration

---

## Machine Learning Workflow

The training process follows these steps:

```text
Dataset
   │
   ▼
Data Validation
   │
   ▼
Duplicate Removal
   │
   ▼
Train/Test Split
   │
   ▼
Preprocessing Pipeline
   │
   ├── Numerical Features
   │
   └── Categorical Features
   │
   ▼
Model Comparison
   │
   ├── Linear Regression
   ├── Random Forest
   ├── Gradient Boosting
   └── Extra Trees
   │
   ▼
5-Fold Cross-Validation
   │
   ▼
Best Model Selection
   │
   ▼
Final Test-Set Evaluation
   │
   ├── MAE
   ├── RMSE
   └── R²
   │
   ▼
Saved Pipeline + Metrics
````

---

## Data Validation

Before training begins, the dataset is checked for common problems.

The training pipeline verifies that:

* All required columns are present.
* The dataset is not empty.
* Required fields do not contain missing values.
* Price values are greater than zero.
* Numerical features contain numeric values.
* Removing duplicate rows does not leave an empty dataset.

Invalid data causes training to stop with a descriptive `ValueError` instead of producing an unreliable model.

---

## Models

The project compares four regression algorithms:

### Linear Regression

A simple baseline regression model used to establish a reference point for the other models.

### Random Forest Regressor

An ensemble of decision trees that can model nonlinear relationships between property features and price.

### Gradient Boosting Regressor

A sequential ensemble method that builds models to correct errors made by previous models.

### Extra Trees Regressor

An ensemble tree method that introduces additional randomization when constructing decision trees.

---

## Model Selection

Each candidate model is evaluated using **5-fold cross-validation** on the training portion of the dataset.

The following metrics are calculated:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination

The model with the highest mean cross-validation **R²** is selected.

The selected model is then trained on the complete training split and evaluated once on the held-out test set.

---

## Current Model Results

The current training run produced the following cross-validation results:

| Model             |     CV MAE |      CV RMSE |  CV R² |
| ----------------- | ---------: | -----------: | -----: |
| Linear Regression | 748,438.39 | 1,021,776.37 | 0.6470 |
| Random Forest     | 767,428.28 | 1,096,284.06 | 0.5975 |
| Gradient Boosting | 789,783.04 | 1,110,164.40 | 0.5858 |
| Extra Trees       | 791,071.75 | 1,131,005.37 | 0.5697 |

Based on cross-validation R², **Linear Regression** was selected as the current best-performing model.

### Final Test-Set Performance

| Metric |        Value |
| ------ | -----------: |
| MAE    |   970,043.40 |
| RMSE   | 1,324,506.96 |
| R²     |       0.6529 |

These values correspond to the current dataset and training configuration. Re-running the training pipeline may produce different results if the dataset or configuration changes.

The complete results are also stored in:

```text
models/model_metrics.json
```

---

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── models/
│   ├── house_price_pipeline.pkl
│   └── model_metrics.json
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── predictor.py
│   ├── train.py
│   └── train.ipynb
│
├── tests/
│   └── test_model.py
│
├── data.csv
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

### Source Files

**`src/train.py`**

Contains the complete model-training workflow, including:

* Data loading
* Data validation
* Preprocessing
* Model comparison
* Cross-validation
* Model selection
* Test-set evaluation
* Model persistence
* Metrics persistence

**`src/predictor.py`**

Contains reusable prediction functionality:

* Loading the trained model
* Validating prediction input
* Generating predictions

Keeping prediction logic separate from the Streamlit interface makes the functionality easier to test and reuse.

**`src/predict.py`**

Contains the Streamlit user interface. It collects property details from the user and passes them to the reusable prediction functions.

**`tests/test_model.py`**

Contains automated tests covering the training pipeline, data validation, model loading, prediction behavior, and invalid inputs.

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python src/train.py
```

The training script will:

1. Load `data.csv`.
2. Validate the dataset.
3. Remove duplicate rows.
4. Split the data into training and test sets.
5. Compare the available regression models.
6. Perform 5-fold cross-validation.
7. Select the model with the highest CV R².
8. Evaluate the selected model on the test set.
9. Save the trained pipeline.
10. Save the evaluation metrics.

The generated files are:

```text
models/house_price_pipeline.pkl
models/model_metrics.json
```

---

## Run the Streamlit Application

Start the application with:

```bash
streamlit run src/predict.py
```

The application provides fields for the property characteristics used by the trained model.

After entering the property details, click:

```text
Predict House Price
```

The application displays the estimated house price.

The prediction is generated using the persisted machine learning pipeline.

---

## Testing

Run the complete test suite with:

```bash
pytest -q
```

The tests cover:

### Model and pipeline

* Trained model file availability
* Model loading
* Model prediction
* Pipeline construction
* Pipeline fitting
* Pipeline prediction

### Dataset validation

* Required columns
* Empty datasets
* Missing values
* Invalid price values
* Invalid numerical data
* Duplicate handling

### Prediction validation

* Valid prediction input
* Empty prediction input
* Invalid input type
* Missing model file

The test suite is intended to catch both normal functionality problems and common failure cases before changes are merged.

---

## Continuous Integration

The project uses **GitHub Actions** to automatically run the test suite.

The workflow is triggered on:

* Pushes
* Pull requests

The CI workflow:

1. Checks out the repository.
2. Sets up Python.
3. Installs dependencies.
4. Runs the pytest test suite.

This helps ensure that changes do not introduce regressions.

---

## Technologies

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Joblib

### Web Application

* Streamlit

### Testing

* Pytest

### CI/CD

* GitHub Actions

### Development

* Jupyter Notebook

---

## Design Decisions

### Reusable preprocessing pipeline

Preprocessing and the regression model are combined into a single scikit-learn `Pipeline`.

This ensures that the same preprocessing steps used during training are automatically applied during prediction.

### Unknown categorical values

The categorical preprocessing uses:

```python
handle_unknown="ignore"
```

This prevents prediction from failing solely because a previously unseen categorical value is encountered.

### Persisted model

The complete trained pipeline is saved rather than only the regression estimator.

This allows the prediction application to load the model and use the same preprocessing configuration that was established during training.

### Separate prediction module

Prediction functionality is kept separate from the Streamlit interface.

This makes the core prediction logic independently testable and allows it to be reused by another interface in the future.

### Data validation

Training validates the input dataset before model fitting.

Failing early on invalid data is preferable to silently training a model using corrupted or incomplete data.

---

## Limitations

This project is intended as a machine learning application and demonstration rather than a production real-estate valuation system.

Important limitations include:

* The model depends on the quality and coverage of the supplied dataset.
* The dataset may not represent current market conditions.
* Predictions should not be treated as guaranteed market prices.
* Model performance may change when new data is introduced.
* No external real-time property or market information is currently used.
* The current model comparison is limited to four regression algorithms.

---

## Future Improvements

Potential improvements include:

* Hyperparameter tuning
* Additional feature engineering
* Expanded model comparison
* Model explainability
* Prediction uncertainty or confidence ranges
* Additional dataset validation
* More extensive test coverage
* Prediction and model-performance visualizations
* Improved deployment configuration
* Monitoring model performance when new data becomes available

These are future possibilities rather than requirements for the current implementation.

---

## Disclaimer

The predicted price is a machine learning estimate based on the supplied dataset and model.

It should not be considered a guaranteed market value, financial advice, or a professional property valuation.

EOF

```