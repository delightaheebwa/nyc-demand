# nyc-demand

Analysis and modelling of the patterns in the demand for NYC Yellow taxis.

A data science project that investigates NYC Yellow Taxi trip patterns — from data acquisition and cleaning through exploratory analysis to predictive modeling — wrapped in an interactive Streamlit web application.

---

## Features

- **Data Pipeline** — Downloads 12 months of NYC TLC parquet data, validates schemas, combines via DuckDB, and cleans invalid records
- **Exploratory Analysis** — Distributions of passenger counts, trip distances, fares, and temporal patterns
- **Predictive Model** — Logistic Regression that predicts whether a trip will have multiple passengers (2+) using temporal and spatial features
- **Interactive App** — Streamlit UI for real-time predictions with cross-version-compatible model artifacts

---

## How It Works

```
NYC TLC Parquet Data
        |
    cleaning.ipynb      Data acquisition, validation, deduplication
        |
    analysis.ipynb      EDA, summary stats, visualizations
        |
    train_model.py      Feature engineering + Logistic Regression
        |
    app.py              Streamlit web app
```

The model uses a scikit-learn Pipeline with:
- **Numeric features:** Cyclical encoding of pickup hour (`sin`/`cos`)
- **Categorical features:** Day of week, weekend flag, pickup location zone (PULocationID)
- **Classifier:** Logistic Regression with `class_weight='balanced'` to handle class imbalance

---

## Repository Structure

```
nyc-demand/
  app.py                          Streamlit web application
  train_model.py                  Model training script
  requirements.txt                Pinned Python dependencies
  .gitignore                      Excludes data/, venv, checkpoints
  notebooks/
    cleaning.ipynb                Data acquisition and cleaning
    analysis.ipynb                Exploratory data analysis
    model_code.ipynb              Model development (Colab)
    model_run.ipynb               Model development (local)
  learnings/
    notes/highlights.md           Development notes and decisions
    img/                          Evaluation screenshots
  data/                           (gitignored) Parquet files and artifacts
```

---

## Installation

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/delightaheebwa/nyc-demand.git
cd nyc-demand

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Data Acquisition & Cleaning

Run the cleaning notebook to download and prepare the data:

```bash
jupyter notebook notebooks/cleaning.ipynb
```

This downloads 12 months of NYC Yellow Taxi parquet data (2025) from the official TLC URLs, combines them, and cleans invalid records.

### 2. Exploratory Analysis

```bash
jupyter notebook notebooks/analysis.ipynb
```

### 3. Train the Model

```bash
python train_model.py
```

This samples 5M rows, engineers features (pickup hour, day of week, weekend flag, cyclical hour encoding, pickup location), and trains a Logistic Regression model. Artifacts are saved to `data/processed/model_artifacts/`.

### 4. Launch the Web App

```bash
streamlit run app.py
```

The app provides a simple interface to predict whether a taxi trip will be multi-passenger based on pickup time and location.

---

## Code Examples

### Running predictions programmatically

```python
import joblib
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Load artifacts
preprocessor = joblib.load("data/processed/model_artifacts/preprocessor.joblib")
with open("data/processed/model_artifacts/model_coefficients.json") as f:
    coefs = json.load(f)

coef_ = np.array(coefs["coef_"])
intercept_ = np.array(coefs["intercept_"])

# Prepare input (e.g., Friday 6 PM from JFK airport zone)
input_df = pd.DataFrame([{
    "pickup_hour_sin": np.sin(2 * np.pi * 18 / 24.0),
    "pickup_hour_cos": np.cos(2 * np.pi * 18 / 24.0),
    "pickup_dow": 5,           # Friday
    "is_weekend": 0,
    "PULocationID": 132,       # JFK Airport
}])

# Predict
X = preprocessor.transform(input_df)
logit = X @ coef_.T + intercept_
proba = 1.0 / (1.0 + np.exp(-logit)).item()
print(f"Multi-passenger probability: {proba:.2%}")
```

### Training a model with custom parameters

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import polars as pl

# Feature engineering
df = pl.read_parquet("data/processed/cleaned_yellow_taxi.parquet") \
    .with_columns([
        (pl.col("passenger_count") >= 2).cast(pl.Int8).alias("is_multi"),
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_dow"),
    ]) \
    .with_columns([
        (pl.col("pickup_dow").is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
    ])

# Build pipeline
preprocess = ColumnTransformer([
    ("num", StandardScaler(), ["pickup_hour_sin", "pickup_hour_cos"]),
    ("cat", OneHotEncoder(drop="first"), ["pickup_dow", "is_weekend", "PULocationID"]),
])

clf = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
])

clf.fit(X, y)
```

---

## Dataset

The data comes from the **NYC Taxi & Limousine Commission (TLC)** Trip Record Data:

- [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- 12 months of Yellow Taxi parquet files (2025)
- ~5M rows sampled for model training

---

## Model Performance

| Metric | Value |
|---|---|
| ROC AUC | ~0.613 |
| PR AUC | ~0.267 |
| Classifier | Logistic Regression (balanced) |

The model is interpretable: coefficients are saved as JSON for cross-version compatibility, allowing the Streamlit app to compute predictions without exact scikit-learn version matching.

---

## License

MIT
