import numpy as np
import joblib
import json
import duckdb
import polars as pl
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent
ANALYZED_PATH = str(PROJECT_ROOT / "data/processed/analyzed_yellow_taxi.parquet")
ARTIFACT_DIR = PROJECT_ROOT / "data/processed/model_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 5_000_000
RANDOM_STATE = 42

print("Sampling 5M rows directly from parquet via DuckDB...")
con = duckdb.connect()

df_sample = con.execute(f"""
    SELECT * FROM read_parquet('{ANALYZED_PATH}')
    USING SAMPLE {SAMPLE_SIZE}
""").pl()

con.close()
print(f"Sampled: {df_sample.height} rows")

print("Feature engineering...")
df_pl = df_sample.with_columns([
    (pl.col("passenger_count") >= 2).cast(pl.Int8).alias("is_multi"),
    pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
    pl.col("tpep_pickup_datetime").dt.weekday().alias("pickup_dow"),
]).with_columns([
    (pl.col("pickup_dow").is_in([6, 7])).cast(pl.Int8).alias("is_weekend"),
    (pl.col("pickup_hour").cast(pl.Float32)
     .map_elements(lambda h: np.sin(2 * np.pi * h / 24.0))
     .alias("pickup_hour_sin")),
    (pl.col("pickup_hour").cast(pl.Float32)
     .map_elements(lambda h: np.cos(2 * np.pi * h / 24.0))
     .alias("pickup_hour_cos")),
])

feature_cols = [
    "pickup_hour_sin", "pickup_hour_cos",
    "pickup_dow", "is_weekend",
    "PULocationID",
]
target_col = "is_multi"

valid_pulocation_ids = (
    df_pl.select("PULocationID")
    .unique()
    .sort("PULocationID")
    .to_series()
    .to_list()
)
with open(ARTIFACT_DIR / "valid_pulocation_ids.json", "w") as f:
    json.dump(valid_pulocation_ids, f)
print(f"Saved {len(valid_pulocation_ids)} valid PULocationIDs")

print("Converting to pandas...")
df_small = df_pl.select(feature_cols + [target_col]).to_pandas()

X = df_small[feature_cols]
y = df_small[target_col]
del df_pl, df_sample, df_small

numeric_cols = ["pickup_hour_sin", "pickup_hour_cos"]
categorical_cols = ["pickup_dow", "is_weekend", "PULocationID"]

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

log_reg = LogisticRegression(
    l1_ratio=0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", log_reg),
])

print("Training model...")
clf.fit(X, y)
print("Training complete.")

joblib.dump(clf, ARTIFACT_DIR / "logreg_pipeline.joblib")

print("Extracting model components for cross-version compatibility...")
preprocessor = clf.named_steps["preprocess"]
model = clf.named_steps["model"]

joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")

model_components = {
    "coef_": model.coef_.tolist(),
    "intercept_": model.intercept_.tolist(),
    "classes_": model.classes_.tolist(),
}
with open(ARTIFACT_DIR / "model_coefficients.json", "w") as f:
    json.dump(model_components, f)

print("All artifacts saved.")
