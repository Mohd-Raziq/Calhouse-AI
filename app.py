import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(BASE_DIR, "pipeline.pkl")

if not (os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE)):
    raise SystemExit(
        "model.pkl / pipeline.pkl not found. Run your training script first "
        "(python train.py) so they are created in this folder."
    )

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# The column names must match the ones used when the pipeline was fitted.
NUMERIC = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
FEATURES = NUMERIC + ["ocean_proximity"]
OPTIONAL = {"total_bedrooms"}  # the pipeline's imputer fills this in when missing
PRED_COLUMN = "predicted_median_house_value"
MAX_ROWS = 20000

# Valid ocean_proximity values, read from the fitted encoder.
OCEAN_VALUES = set(
    pipeline.named_transformers_["cat"].named_steps["one_hot"].categories_[0]
)

app = Flask(__name__)


def to_matrix(df):
    """Run the saved preprocessing pipeline and return a dense array."""
    X = pipeline.transform(df[FEATURES])
    return X.toarray() if hasattr(X, "toarray") else X


@app.get("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object with the housing features."), 400

    row = {}
    for name in NUMERIC:
        value = payload.get(name)
        if value is None or value == "":
            if name in OPTIONAL:
                row[name] = np.nan
                continue
            return jsonify(error=f"'{name}' is required."), 400
        try:
            number = float(value)
        except (TypeError, ValueError):
            return jsonify(error=f"'{name}' must be a number."), 400
        if not math.isfinite(number):
            return jsonify(error=f"'{name}' must be a finite number."), 400
        row[name] = number

    ocean = payload.get("ocean_proximity")
    if ocean not in OCEAN_VALUES:
        options = ", ".join(sorted(OCEAN_VALUES))
        return jsonify(error=f"'ocean_proximity' must be one of: {options}."), 400
    row["ocean_proximity"] = ocean

    X = to_matrix(pd.DataFrame([row]))
    estimate = float(model.predict(X)[0])

    # Spread across the forest's individual trees gives a rough range.
    tree_predictions = np.array([tree.predict(X)[0] for tree in model.estimators_])
    low, high = np.percentile(tree_predictions, [10, 90])

    return jsonify(
        prediction=round(estimate, 2),
        low=round(float(low), 2),
        high=round(float(high), 2),
    )


@app.post("/api/predict-csv")
def predict_csv():
    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return jsonify(error="Choose a CSV file to upload."), 400

    try:
        df = pd.read_csv(upload)
    except Exception:
        return jsonify(error="That file could not be read as a CSV."), 400

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return jsonify(error=f"The CSV is missing these columns: {', '.join(missing)}."), 400
    if df.empty:
        return jsonify(error="The CSV has no rows."), 400
    if len(df) > MAX_ROWS:
        return jsonify(error=f"Please upload at most {MAX_ROWS:,} rows at a time."), 400

    unknown = sorted(set(df["ocean_proximity"].dropna().unique()) - OCEAN_VALUES)
    if unknown or df["ocean_proximity"].isna().any():
        options = ", ".join(sorted(OCEAN_VALUES))
        return jsonify(error=f"'ocean_proximity' must be one of: {options}."), 400

    for name in NUMERIC:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    df[PRED_COLUMN] = np.round(model.predict(to_matrix(df)), 2)

    # to_json turns NaN into null, which is valid JSON.
    rows = json.loads(df.to_json(orient="records"))
    return jsonify(count=len(df), columns=list(df.columns), rows=rows)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
