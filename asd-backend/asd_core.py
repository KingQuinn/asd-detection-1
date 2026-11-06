"""Shared feature definitions and inference helpers for the ASD detection model.

Used by the training script (train.py), the Flask API (app.py) and the
Streamlit app, so preprocessing at prediction time always matches training.
"""
import os
import pickle

import pandas as pd

AQ_ITEMS = [f"A{i}" for i in range(1, 11)]
# Column order the model was trained on. "Jauundice" keeps the dataset's spelling.
FEATURES = AQ_ITEMS + ["Age", "Sex", "Jauundice", "Family_ASD"]

# Binary categorical encodings (identical to sklearn LabelEncoder's alphabetical order).
ENCODINGS = {
    "Sex": {"f": 0, "m": 1},
    "Jauundice": {"no": 0, "yes": 1},
    "Family_ASD": {"no": 0, "yes": 1},
}

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asd_detection2.pkl")


class InputError(ValueError):
    """Raised when a prediction request is missing fields or has invalid values."""


def load_model(path=DEFAULT_MODEL_PATH):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if "features" not in bundle:
        raise RuntimeError(
            f"{path} was produced by the old training scripts and is not compatible. "
            "Re-run `python train.py` from the project root."
        )
    return bundle


def prepare_input(payload):
    """Validate a raw request dict and return a one-row DataFrame in training order.

    Accepts "Jaundice" (API spelling) or "Jauundice" (dataset spelling).
    """
    payload = dict(payload)
    if "Jauundice" not in payload and "Jaundice" in payload:
        payload["Jauundice"] = payload["Jaundice"]

    missing = [f.replace("Jauundice", "Jaundice") for f in FEATURES if payload.get(f) in (None, "")]
    if missing:
        raise InputError(f"Missing fields: {', '.join(missing)}")

    row = {}
    for item in AQ_ITEMS:
        try:
            value = int(payload[item])
        except (TypeError, ValueError):
            raise InputError(f"{item} must be 0 or 1")
        if value not in (0, 1):
            raise InputError(f"{item} must be 0 or 1")
        row[item] = value

    try:
        age = float(payload["Age"])
    except (TypeError, ValueError):
        raise InputError("Age must be a number")
    if not 1 <= age <= 100:
        raise InputError("Age must be between 1 and 100")
    row["Age"] = age

    for col, mapping in ENCODINGS.items():
        value = str(payload[col]).strip().lower()
        if value not in mapping:
            raise InputError(f"{col} must be one of {sorted(mapping)}")
        row[col] = mapping[value]

    return pd.DataFrame([row], columns=FEATURES)


def get_supabase():
    """Return a Supabase client if SUPABASE_URL and SUPABASE_KEY are set, else None."""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        return None
    from supabase import create_client

    try:
        return create_client(url, key)
    except Exception as e:
        # Bad credentials should disable saving, not take the app down.
        print(f"Supabase disabled: {e}")
        return None


def save_prediction(client, payload, label):
    """Insert one screening record into the Supabase "predictions" table."""
    record = {item: int(payload[item]) for item in AQ_ITEMS}
    record.update({
        "Age": int(float(payload["Age"])),
        "Sex": str(payload["Sex"]).lower(),
        "Jaundice": str(payload.get("Jaundice", payload.get("Jauundice"))).lower(),
        "Family_ASD": str(payload["Family_ASD"]).lower(),
        "Prediction": "yes" if label == 1 else "no",
    })
    client.table("predictions").insert(record).execute()


def predict(bundle, payload):
    """Return (label, probability_of_asd) for one screening record."""
    X = prepare_input(payload)
    proba = float(bundle["model"].predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    return label, proba
