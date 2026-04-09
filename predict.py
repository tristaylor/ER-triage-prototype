"""
predict.py
──────────
Loads a trained XGBoost model and produces a deterioration risk score
(0–1) for one or more patients.

Can be used:
  1. Batch mode — score a whole CSV of patients
  2. Single patient — pass a dict of features and get back a score

Usage (batch):
  python src/predict.py --model models/xgb_model.pkl \
                        --features data/features.csv \
                        --output data/predictions.csv
"""

import os
import argparse
import logging
import json
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ID_COLS = ["stay_id", "subject_id", "hadm_id"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """Load a serialised joblib model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    log.info("Loaded model from %s", model_path)
    return model


def load_feature_names(model_dir: str) -> list[str]:
    """Load the ordered list of feature names used at training time."""
    path = os.path.join(model_dir, "feature_names.json")
    with open(path) as f:
        return json.load(f)


# ── Prediction helpers ────────────────────────────────────────────────────────

def align_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Ensure the input DataFrame has exactly the columns the model expects,
    in the right order.  Missing columns are filled with 0.
    """
    missing = set(feature_names) - set(df.columns)
    if missing:
        log.warning("Missing %d feature(s); filling with 0: %s",
                    len(missing), sorted(missing)[:5])
    for col in missing:
        df[col] = 0
    return df[feature_names]


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return probability of deterioration (class 1) for each row."""
    return model.predict_proba(X)[:, 1]


def risk_tier(score: float) -> str:
    """Convert a continuous score to a human-readable risk tier."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.25:
        return "MODERATE"
    return "LOW"


# ── Public API ────────────────────────────────────────────────────────────────

def score_dataframe(model,
                    feature_names: list[str],
                    df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of patients.
    Returns the original DataFrame with 'risk_score' and 'risk_tier' columns.
    """
    ids = df[[c for c in ID_COLS if c in df.columns]].copy()
    X   = align_features(df.copy(), feature_names)

    probs = predict_proba(model, X)
    result = ids.copy()
    result["risk_score"] = np.round(probs, 4)
    result["risk_tier"]  = [risk_tier(p) for p in probs]
    return result


def score_patient(model,
                  feature_names: list[str],
                  patient: dict) -> dict:
    """
    Score a single patient represented as a feature dictionary.
    Returns a dict with risk_score and risk_tier.

    Example:
        score_patient(model, feature_names, {
            "heart_rate_mean": 112,
            "sbp_mean": 88,
            "spo2_mean": 91,
            ...
        })
    """
    df = pd.DataFrame([patient])
    result = score_dataframe(model, feature_names, df)
    return {
        "risk_score": float(result["risk_score"].iloc[0]),
        "risk_tier":  result["risk_tier"].iloc[0],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run(model_path: str,
        features_path: str,
        output_path: str) -> pd.DataFrame:

    model_dir     = os.path.dirname(model_path)
    model         = load_model(model_path)
    feature_names = load_feature_names(model_dir)

    df     = pd.read_csv(features_path, low_memory=False)
    result = score_dataframe(model, feature_names, df)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Scored %d patients → %s", len(result), output_path)

    tier_counts = result["risk_tier"].value_counts()
    log.info("Risk tier distribution:\n%s", tier_counts.to_string())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/xgb_model.pkl")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--output",   default="data/predictions.csv")
    args = parser.parse_args()
    run(args.model, args.features, args.output)
