# Iris Prediction

A machine learning project that classifies Iris flowers by species from four sepal and petal measurements. It includes a Streamlit interface for interactive predictions and an MLflow-tracked training pipeline.

## Features

- Predicts an Iris species from sepal length, sepal width, petal length, and petal width.
- Provides an interactive Streamlit web interface.
- Runs a data ingestion, preprocessing, model training, and evaluation pipeline.
- Tracks model parameters and evaluation metrics with MLflow.

## Model

The training pipeline uses a `StandardScaler` followed by a `RandomForestClassifier` with Gini criterion and maximum depth 4. The pipeline evaluates accuracy, macro precision, and macro recall. A model is approved by the pipeline when its accuracy is at least 0.90.

## Project Structure

```text
.
├── app.py                    # Streamlit prediction interface
├── IRIS.csv                  # Source dataset used by data ingestion
├── train.csv                 # Additional training data file
├── test.csv                  # Additional test data file
├── data_ingestion.py         # Copies and validates the source dataset
├── pre_processing.py         # Train/test split and feature scaling
├── train.py                  # Trains and logs the classifier with MLflow
├── evaluation.py             # Computes and logs evaluation metrics
├── pipeline.py               # Runs the end-to-end training pipeline
├── requirements.txt          # Python dependencies
├── mlflow.db                 # Local MLflow tracking database
└── artifacts/
    ├── preprocessor.pkl      # Saved feature scaler
    └── model.pkl             # Saved model used by the app
```

## Requirements

- Python 3.9 or later is recommended.
- Packages listed in `requirements.txt`.

## Run the Web App

Open a terminal in the project directory and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies and launch Streamlit:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Run the Training Pipeline

The pipeline reads `IRIS.csv`, creates an `ingested/` directory, splits the data into training and test sets, scales the features, trains and evaluates a Random Forest classifier, and logs the run to MLflow.

```bash
python pipeline.py
```

The pipeline saves the preprocessor to `artifacts/preprocessor.pkl`. The Streamlit app expects both `artifacts/preprocessor.pkl` and `artifacts/model.pkl` to exist in the project directory.

To inspect MLflow runs locally, use:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the URL shown by MLflow, usually `http://localhost:5000`.

## Dataset

`IRIS.csv` is the source dataset used by the training pipeline. The pipeline expects a `species` target column and four numeric measurement columns. `train.csv` and `test.csv` are also present in the repository as separate dataset files.

## Notes

- Keep the files in `artifacts/` in place when running the app.
- Retraining runs the training pipeline and logs the model to MLflow. The current training script logs the model to MLflow; ensure `artifacts/model.pkl` is available for the app after training.
- The project is intended for learning and demonstration purposes.
