"""
features.py
───────────
Takes the cleaned vitals DataFrame (output of preprocess.py) and engineers
clinically meaningful features for the deterioration model.

Engineered features include:
  - Shock Index (HR / SBP)
  - Pulse Pressure (SBP - DBP)
  - MAP (mean arterial pressure)
  - Low SpO2 flag
  - Tachycardia / bradycardia / hypotension flags
  - Rolling statistics (mean, std, min/max) over a configurable window
  - Rate-of-change (delta) for key vitals

Usage:
  python src/features.py --input data/processed/vitals_clean.parquet \
                         --output data/features.csv \
                         --window_hours 2
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

VITALS = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temp_c"]


# ── Point-in-time features ────────────────────────────────────────────────────

def add_derived_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Clinically standard composite vital signs."""
    df = df.copy()

    # Shock Index: HR / SBP — values > 1.0 associated with haemodynamic instability
    df["shock_index"] = np.where(
        df["sbp"] > 0, df["heart_rate"] / df["sbp"], np.nan
    )

    # Pulse Pressure: widens in sepsis, narrows in tamponade/shock
    df["pulse_pressure"] = df["sbp"] - df["dbp"]

    # Mean Arterial Pressure: (SBP + 2*DBP) / 3 — perfusion proxy
    df["map"] = (df["sbp"] + 2 * df["dbp"]) / 3

    return df


def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flags for clinically significant thresholds."""
    df = df.copy()

    df["flag_tachycardia"]  = (df["heart_rate"] > 100).astype(int)
    df["flag_bradycardia"]  = (df["heart_rate"] < 60).astype(int)
    df["flag_hypotension"]  = (df["sbp"] < 90).astype(int)
    df["flag_hypertension"] = (df["sbp"] > 180).astype(int)
    df["flag_tachypnea"]    = (df["resp_rate"] > 20).astype(int)
    df["flag_low_spo2"]     = (df["spo2"] < 94).astype(int)
    df["flag_fever"]        = (df["temp_c"] > 38.3).astype(int)
    df["flag_hypothermia"]  = (df["temp_c"] < 36.0).astype(int)

    # SIRS criteria count (0–4): used as severity proxy
    df["sirs_count"] = (
        df["flag_tachycardia"] +
        df["flag_tachypnea"] +
        df["flag_fever"] +
        df["flag_hypothermia"]
    )

    return df


# ── Rolling / temporal features ───────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, window_hours: float = 2.0) -> pd.DataFrame:
    """
    For each stay, compute rolling statistics over the past `window_hours`
    hours for key vitals.

    The DataFrame must be sorted by (stay_id, charttime) before calling.
    """
    df = df.sort_values(["stay_id", "charttime"]).copy()
    df = df.set_index("charttime")

    roll_vitals = ["heart_rate", "sbp", "resp_rate", "spo2"]
    window = f"{int(window_hours * 60)}min"

    results = []
    for stay_id, group in df.groupby("stay_id"):
        g = group.copy()
        for vital in roll_vitals:
            if vital not in g.columns:
                continue
            rolled = g[vital].rolling(window, min_periods=1)
            g[f"{vital}_roll_mean"] = rolled.mean()
            g[f"{vital}_roll_std"]  = rolled.std().fillna(0)
            g[f"{vital}_roll_min"]  = rolled.min()
            g[f"{vital}_roll_max"]  = rolled.max()
        results.append(g)

    out = pd.concat(results).reset_index()
    log.info("Rolling features added (window=%s)", window)
    return out


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rate-of-change: difference between current value and previous observation
    within the same stay, for key vitals.
    """
    df = df.sort_values(["stay_id", "charttime"]).copy()
    delta_vitals = ["heart_rate", "sbp", "resp_rate", "spo2"]

    for vital in delta_vitals:
        if vital not in df.columns:
            continue
        df[f"{vital}_delta"] = df.groupby("stay_id")[vital].diff().fillna(0)

    log.info("Delta features added")
    return df


# ── Missing-value handling ────────────────────────────────────────────────────

def impute_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill within each stay (carries last observation forward), then
    fill remaining NaNs with the population median.
    """
    df = df.sort_values(["stay_id", "charttime"]).copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ["subject_id", "hadm_id", "stay_id", "sex_male",
               "hospital_expire_flag", "hours_since_icu_admit"]
    fill_cols = [c for c in numeric_cols if c not in exclude]

    # Forward-fill within stay
    df[fill_cols] = df.groupby("stay_id")[fill_cols].ffill()

    # Fill residual NaNs with population median
    medians = df[fill_cols].median()
    df[fill_cols] = df[fill_cols].fillna(medians)

    log.info("Imputation complete. Remaining NaNs: %d",
             df[fill_cols].isna().sum().sum())
    return df


# ── Snapshot aggregation ──────────────────────────────────────────────────────

def build_admission_snapshot(df: pd.DataFrame,
                              first_n_hours: float = 2.0) -> pd.DataFrame:
    """
    Aggregate all observations within the first `first_n_hours` of ICU
    admission into a single feature vector per stay.  This is the format
    fed to the XGBoost model.
    """
    window = df[df["hours_since_icu_admit"] <= first_n_hours].copy()

    agg_funcs = {col: ["mean", "min", "max", "std"]
                 for col in VITALS if col in window.columns}
    agg_funcs.update({
        "shock_index":     ["mean", "max"],
        "map":             ["mean", "min"],
        "sirs_count":      ["mean", "max"],
        "flag_low_spo2":   ["max"],
        "flag_hypotension":["max"],
        "flag_tachycardia":["max"],
        "hours_since_icu_admit": ["max"],
    })

    # Keep only columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in window.columns}

    snapshot = window.groupby("stay_id").agg(agg_funcs)
    snapshot.columns = ["_".join(c).strip() for c in snapshot.columns]
    snapshot = snapshot.reset_index()

    # Attach static features (take first row per stay — they're constant)
    static = (df.sort_values("charttime")
                .groupby("stay_id")[["subject_id", "hadm_id",
                                     "anchor_age", "sex_male",
                                     "admission_type"]]
                .first()
                .reset_index())

    snapshot = snapshot.merge(static, on="stay_id", how="left")

    # One-hot encode admission_type
    if "admission_type" in snapshot.columns:
        snapshot = pd.get_dummies(snapshot, columns=["admission_type"],
                                  prefix="admtype", dtype=int)

    log.info("Snapshot : %d stays × %d features", *snapshot.shape)
    return snapshot


# ── Main ─────────────────────────────────────────────────────────────────────

def run(input_path: str, output_path: str, window_hours: float = 2.0) -> pd.DataFrame:
    log.info("Loading %s", input_path)
    df = pd.read_parquet(input_path) if input_path.endswith(".parquet") \
         else pd.read_csv(input_path, low_memory=False)

    df = add_derived_vitals(df)
    df = add_flag_features(df)
    df = add_rolling_features(df, window_hours=window_hours)
    df = add_delta_features(df)
    df = impute_and_clip(df)
    snapshot = build_admission_snapshot(df, first_n_hours=window_hours)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    snapshot.to_csv(output_path, index=False)
    log.info("Saved → %s", output_path)
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="data/processed/vitals_clean.parquet")
    parser.add_argument("--output",       default="data/features.csv")
    parser.add_argument("--window_hours", type=float, default=2.0,
                        help="Hours of data used for the admission snapshot")
    args = parser.parse_args()
    run(args.input, args.output, args.window_hours)
