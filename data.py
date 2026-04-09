"""
preprocess.py
─────────────
Loads and cleans raw MIMIC-IV tables (chartevents, admissions, patients,
icustays) and produces a single tidy vitals DataFrame ready for feature
engineering.

Expected raw files (CSV or Parquet) under data/raw/:
  - patients.csv
  - admissions.csv
  - chartevents.csv      ← large; Parquet strongly recommended
  - icustays.csv

Usage:
  python src/preprocess.py --input data/raw/ --output data/processed/
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── MIMIC-IV itemids for the vitals we care about ────────────────────────────
VITAL_ITEM_IDS = {
    "heart_rate":   [220045],
    "sbp":          [220179, 220050],   # non-invasive + arterial
    "dbp":          [220180, 220051],
    "resp_rate":    [220210],
    "spo2":         [220277],
    "temp_c":       [223762],
    "temp_f":       [223761],
}

# Flatten to a lookup dict  itemid → vital_name
ITEMID_MAP = {iid: name for name, ids in VITAL_ITEM_IDS.items() for iid in ids}

# Plausible physiologic ranges for outlier removal
VITAL_BOUNDS = {
    "heart_rate": (10,  300),
    "sbp":        (40,  300),
    "dbp":        (10,  200),
    "resp_rate":  (4,   60),
    "spo2":       (50,  100),
    "temp_c":     (25,  45),
    "temp_f":     (77,  113),
}


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load(path: str, **kwargs) -> pd.DataFrame:
    """Auto-detect CSV vs Parquet and load."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_patients(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "patients.csv")
    df = _load(path, usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
    log.info("patients   : %d rows", len(df))
    return df


def load_admissions(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "admissions.csv")
    df = _load(path, usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime",
        "deathtime", "hospital_expire_flag", "admission_type",
        "race", "insurance",
    ])
    for col in ["admittime", "dischtime", "deathtime"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    log.info("admissions : %d rows", len(df))
    return df


def load_icustays(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "icustays.csv")
    df = _load(path, usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"])
    df["intime"]  = pd.to_datetime(df["intime"],  errors="coerce")
    df["outtime"] = pd.to_datetime(df["outtime"], errors="coerce")
    log.info("icustays   : %d rows", len(df))
    return df


def load_chartevents(raw_dir: str) -> pd.DataFrame:
    """
    Load only the itemids we need.  chartevents is huge (~330 M rows);
    filtering on read cuts memory dramatically.
    """
    target_ids = list(ITEMID_MAP.keys())
    path = os.path.join(raw_dir, "chartevents.csv")
    if path.endswith(".parquet") or os.path.exists(path.replace(".csv", ".parquet")):
        pq_path = path.replace(".csv", ".parquet")
        df = pd.read_parquet(pq_path, columns=["subject_id", "hadm_id", "stay_id",
                                                "charttime", "itemid", "valuenum"])
        df = df[df["itemid"].isin(target_ids)]
    else:
        chunks = []
        for chunk in pd.read_csv(path, usecols=["subject_id", "hadm_id", "stay_id",
                                                  "charttime", "itemid", "valuenum"],
                                  chunksize=500_000, low_memory=False):
            chunks.append(chunk[chunk["itemid"].isin(target_ids)])
        df = pd.concat(chunks, ignore_index=True)

    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    log.info("chartevents (filtered): %d rows", len(df))
    return df


# ── Cleaning ─────────────────────────────────────────────────────────────────

def clean_chartevents(df: pd.DataFrame) -> pd.DataFrame:
    """Map itemids → vital names, remove outliers, unify temperature to °C."""
    df = df.copy()
    df["vital"] = df["itemid"].map(ITEMID_MAP)
    df = df.dropna(subset=["vital", "valuenum", "charttime"])

    # Remove physiologically impossible values
    def _filter(group):
        name = group["vital"].iloc[0]
        lo, hi = VITAL_BOUNDS[name]
        return group[(group["valuenum"] >= lo) & (group["valuenum"] <= hi)]

    df = df.groupby("vital", group_keys=False).apply(_filter)

    # Convert °F → °C, then drop temp_f
    mask_f = df["vital"] == "temp_f"
    df.loc[mask_f, "vital"]    = "temp_c"
    df.loc[mask_f, "valuenum"] = (df.loc[mask_f, "valuenum"] - 32) * 5 / 9

    # Keep one row per (stay_id, charttime, vital) — take the mean if duplicates
    df = (df.groupby(["subject_id", "hadm_id", "stay_id", "charttime", "vital"],
                     as_index=False)["valuenum"]
            .mean())

    log.info("chartevents (clean) : %d rows", len(df))
    return df


def pivot_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot from long format (one row per vital measurement) to wide format
    (one row per stay_id + charttime, one column per vital).
    """
    wide = df.pivot_table(
        index=["subject_id", "hadm_id", "stay_id", "charttime"],
        columns="vital",
        values="valuenum",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    log.info("vitals wide : %d rows × %d cols", *wide.shape)
    return wide


# ── Patient demographics ──────────────────────────────────────────────────────

def merge_demographics(vitals: pd.DataFrame,
                       patients: pd.DataFrame,
                       admissions: pd.DataFrame,
                       icustays: pd.DataFrame) -> pd.DataFrame:
    """
    Attach age, sex, admission metadata, and hours-since-admission to the
    vitals table.
    """
    # anchor_age in MIMIC-IV is age at anchor_year admission
    df = vitals.merge(patients[["subject_id", "gender", "anchor_age"]],
                      on="subject_id", how="left")
    df = df.merge(admissions[["hadm_id", "admittime", "dischtime",
                               "hospital_expire_flag", "admission_type"]],
                  on="hadm_id", how="left")
    df = df.merge(icustays[["stay_id", "intime"]],
                  on="stay_id", how="left")

    # Hours elapsed since ICU admission at time of each vital measurement
    df["hours_since_icu_admit"] = (
        (df["charttime"] - df["intime"]).dt.total_seconds() / 3600
    )

    # Drop rows recorded before ICU admission (data artefacts)
    df = df[df["hours_since_icu_admit"] >= 0]

    # Encode sex as binary
    df["sex_male"] = (df["gender"] == "M").astype(int)
    df = df.drop(columns=["gender"])

    log.info("merged demographics : %d rows", len(df))
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def run(raw_dir: str, output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    patients    = load_patients(raw_dir)
    admissions  = load_admissions(raw_dir)
    icustays    = load_icustays(raw_dir)
    chartevents = load_chartevents(raw_dir)

    clean   = clean_chartevents(chartevents)
    wide    = pivot_vitals(clean)
    merged  = merge_demographics(wide, patients, admissions, icustays)

    out_path = os.path.join(output_dir, "vitals_clean.parquet")
    merged.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/",       help="Raw MIMIC-IV directory")
    parser.add_argument("--output", default="data/processed/", help="Output directory")
    args = parser.parse_args()
    run(args.input, args.output)
