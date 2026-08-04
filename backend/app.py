"""SuperKart sales forecasting API.

Exposes a single-record endpoint and a batch endpoint over a serialised
scikit-learn pipeline that carries its own preprocessing.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

superkart_sales_api = Flask("SuperKart Sales Forecast")

MODEL_PATH = os.environ.get("MODEL_PATH", "superkart_model.joblib")
METADATA_PATH = os.environ.get("METADATA_PATH", "model_metadata.json")

model = joblib.load(MODEL_PATH)

try:
    with open(METADATA_PATH) as fh:
        MODEL_METADATA = json.load(fh)
except FileNotFoundError:
    MODEL_METADATA = {}

# The feature contract. This must match X_train.columns from the training
# notebook exactly, in both membership and order.
EXPECTED_FEATURE_COLUMNS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
    "Store_Age_Years",
]

NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

# Known vocabularies. handle_unknown="ignore" would otherwise accept an unknown
# category, encode it as all zeros, and return a confident wrong answer.
ALLOWED_VALUES = {
    "Product_Sugar_Content": {"Low Sugar", "Regular", "No Sugar"},
    "Store_Size": {"Small", "Medium", "High"},
    "Store_Location_City_Type": {"Tier 1", "Tier 2", "Tier 3"},
    "Store_Type": {
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
    },
    "Product_Id_char": {"FD", "DR", "NC"},
    "Product_Type_Category": {"Perishables", "Non Perishables"},
}

# The same relabelling applied during training. Callers using the raw source
# vocabulary would otherwise receive a silently degraded prediction.
VALUE_ALIASES = {"Product_Sugar_Content": {"reg": "Regular"}}


class ValidationError(Exception):
    """Raised when caller input cannot be turned into a valid feature frame."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def prepare_features(frame):
    """Validate, normalise and order a DataFrame so the pipeline can consume it.

    Raises ValidationError with an actionable message on any problem.
    """
    frame = frame.copy()

    missing = [c for c in EXPECTED_FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValidationError(
            "Input is missing required feature columns.",
            {"missing_columns": missing, "expected_columns": EXPECTED_FEATURE_COLUMNS},
        )

    extra = [c for c in frame.columns if c not in EXPECTED_FEATURE_COLUMNS]
    if extra:
        frame = frame.drop(columns=extra)

    # Normalise known aliases before validating vocabularies.
    for column, aliases in VALUE_ALIASES.items():
        frame[column] = frame[column].replace(aliases)

    # Coerce numerics. Anything that will not convert becomes NaN and is rejected.
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    bad_numeric = {
        column: int(frame[column].isna().sum())
        for column in NUMERIC_FEATURES
        if frame[column].isna().any()
    }
    if bad_numeric:
        raise ValidationError(
            "One or more numeric features contain values that are not numbers.",
            {"columns_with_invalid_values": bad_numeric},
        )

    # Validate categorical vocabularies.
    invalid_categories = {}
    for column, allowed in ALLOWED_VALUES.items():
        seen = set(frame[column].dropna().astype(str).unique())
        unknown = sorted(seen - allowed)
        if unknown:
            invalid_categories[column] = {
                "unexpected": unknown,
                "allowed": sorted(allowed),
            }
    if invalid_categories:
        raise ValidationError(
            "One or more categorical features contain unrecognised values.",
            {"invalid_categories": invalid_categories},
        )

    return frame[EXPECTED_FEATURE_COLUMNS]


@superkart_sales_api.get("/")
def home():
    return jsonify(
        {
            "service": "SuperKart Sales Forecast API",
            "endpoints": ["/health", "/v1/sales/single", "/v1/sales/batch"],
            "expected_features": EXPECTED_FEATURE_COLUMNS,
        }
    )


@superkart_sales_api.get("/health")
def health():
    """Liveness and readiness check that confirms the model actually loaded."""
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_name": MODEL_METADATA.get("model_name", "unknown"),
            "expected_features": EXPECTED_FEATURE_COLUMNS,
        }
    )


@superkart_sales_api.post("/v1/sales/single")
def predict_sales():
    """Predict sales for a single product and store combination."""
    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a non-empty JSON object."}), 400

    try:
        features = prepare_features(pd.DataFrame([payload]))
    except ValidationError as exc:
        return jsonify({"error": exc.message, **exc.details}), 400

    try:
        prediction = float(model.predict(features)[0])
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify({"Predicted_Sales": prediction})


@superkart_sales_api.post("/v1/sales/batch")
def predict_sales_batch():
    """Predict sales for a CSV of product and store combinations."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request. Send the CSV under the key 'file'."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        return jsonify({"error": f"Failed to read the CSV file: {exc}"}), 400

    if frame.empty:
        return jsonify({"error": "The uploaded CSV contains no rows."}), 400

    # Keep an identifier for the response if one is supplied, otherwise use the
    # row position. Identifiers are returned in a list, so duplicates survive.
    if "Product_Id" in frame.columns:
        identifiers = frame["Product_Id"].astype(str).tolist()
        frame = frame.drop(columns=["Product_Id"])
    else:
        identifiers = [str(i) for i in range(len(frame))]

    try:
        features = prepare_features(frame)
    except ValidationError as exc:
        return jsonify({"error": exc.message, **exc.details}), 400

    try:
        predictions = model.predict(features).tolist()
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(
        {
            "count": len(predictions),
            "predictions": [
                {"id": identifier, "predicted_sales": float(value)}
                for identifier, value in zip(identifiers, predictions)
            ],
        }
    )


if __name__ == "__main__":
    # Development entrypoint only. Production uses gunicorn, see the Dockerfile.
    # debug is False: the Werkzeug debugger must never run in a container.
    superkart_sales_api.run(debug=False, host="0.0.0.0", port=7860)
