# House Price Prediction

A machine learning project that predicts house prices from property features such as area, bedrooms, bathrooms, location-related attributes, parking, and furnishing status.

The project includes a complete training pipeline, model comparison, automated tests, and a Streamlit web application for predictions.

## Features

* House price prediction using machine learning
* Comparison of multiple regression models
* 5-fold cross-validation for model selection
* Automated preprocessing with scikit-learn
* Categorical feature encoding with `OneHotEncoder`
* Saved end-to-end ML pipeline using `joblib`
* Streamlit prediction interface
* Automated tests with `pytest`
* GitHub Actions CI for every push and pull request

## Machine Learning Workflow

```text
data.csv
   ↓
Data cleaning
   ↓
Train/Test Split
   ↓
5-Fold Cross-Validation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Final Test Evaluation
   ↓
Saved ML Pipeline
   ↓
Streamlit Prediction App
```

## Models Compared

The training pipeline evaluates:

* Linear Regression
* Random Forest
* Gradient Boosting
* Extra Trees

Model selection is based on cross-validation performance on the training data. The held-out test set is used only for final evaluation.

### Current Results

Using the current dataset and a fixed random seed:

| Model             |  CV MAE |   CV RMSE |      CV R² |
| ----------------- | ------: | --------: | ---------: |
| Linear Regression | 748,438 | 1,021,776 | **0.6470** |
| Random Forest     | 767,428 | 1,096,284 |     0.5975 |
| Gradient Boosting | 789,783 | 1,110,164 |     0.5858 |
| Extra Trees       | 791,072 | 1,131,005 |     0.5697 |

Linear Regression currently performs best during cross-validation.

Final held-out test performance:

* **MAE:** 970,043
* **RMSE:** 1,324,507
* **R²:** 0.6529

## Project Structure

```text
VSOC-ACT-2/
├── .github/
│   └── workflows/
│       └── tests.yml
├── models/
│   └── house_price_pipeline.pkl
├── tests/
│   └── test_model.py
├── src/
│   ├── predict.py
│   ├── train.py
│   └── train.ipynb
├── data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/AsamNivedha/VSOC-ACT-2.git
cd VSOC-ACT-2
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python src/train.py
```

The training script:

1. Loads and cleans the dataset.
2. Splits the data into training and test sets.
3. Builds preprocessing pipelines.
4. Compares four regression models using 5-fold cross-validation.
5. Selects the best-performing model.
6. Evaluates it on the held-out test set.
7. Saves the complete preprocessing + model pipeline to `models/house_price_pipeline.pkl`.

## Run the Streamlit App

Start the application with:

```bash
streamlit run src/predict.py
```

The application provides a form where users can enter property details and receive a predicted house price.

## Run Tests

Run the test suite with:

```bash
pytest -q
```

The tests verify:

* The trained model file exists.
* The saved pipeline can generate predictions.
* The dataset contains all required features.

## Continuous Integration

GitHub Actions automatically runs the test suite when changes are pushed or a pull request is opened.

Workflow:

```text
Push / Pull Request
        ↓
Install Python
        ↓
Install dependencies
        ↓
Run pytest
        ↓
Pass / Fail
```

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Pytest
* GitHub Actions

## Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Feature engineering
* Model explainability
* Prediction confidence/range estimates
* Improved Streamlit visualizations
* More comprehensive test coverage

