import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "data/processed/model_artifacts"

@st.cache_resource
def load_preprocessor():
    return joblib.load(ARTIFACT_DIR / "preprocessor.joblib")

@st.cache_data
def load_location_ids():
    with open(ARTIFACT_DIR / "valid_pulocation_ids.json") as f:
        return json.load(f)

@st.cache_data
def load_model_coefficients():
    with open(ARTIFACT_DIR / "model_coefficients.json") as f:
        return json.load(f)

preprocessor = load_preprocessor()
valid_ids = load_location_ids()
model_coefs = load_model_coefficients()

coef_ = np.array(model_coefs["coef_"])
intercept_ = np.array(model_coefs["intercept_"])

st.title("NYC Taxi Multi-Passenger Predictor")

pickup_hour = st.slider("Pickup hour", 0, 23, 12)
dow_label = st.selectbox(
    "Day of week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
pulocation_id = st.selectbox("Pickup location zone ID", valid_ids)

dow_map = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4,
    "Friday": 5, "Saturday": 6, "Sunday": 7,
}
pickup_dow = dow_map[dow_label]
is_weekend = 1 if pickup_dow in (6, 7) else 0

pickup_hour_sin = np.sin(2 * np.pi * pickup_hour / 24.0)
pickup_hour_cos = np.cos(2 * np.pi * pickup_hour / 24.0)

input_df = pd.DataFrame([{
    "pickup_hour_sin": pickup_hour_sin,
    "pickup_hour_cos": pickup_hour_cos,
    "pickup_dow": pickup_dow,
    "is_weekend": is_weekend,
    "PULocationID": int(pulocation_id),
}]).astype({
    "pickup_dow": "int8",
    "is_weekend": "int8",
    "PULocationID": "int32",
})

X_transformed = preprocessor.transform(input_df)
logit = X_transformed @ coef_.T + intercept_
proba = 1.0 / (1.0 + np.exp(-logit)).item()
prediction = "Multi-passenger" if proba >= 0.5 else "Single-passenger"

st.metric("Probability of multi-passenger", f"{proba:.2%}")
st.write(f"**Prediction:** {prediction}")
