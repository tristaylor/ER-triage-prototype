"""
labels.py
─────────
Creates binary deterioration labels for each ICU stay.

A patient is labelled as "deteriorated" (label = 1) if ANY of the following
occur within `window_hours` of ICU admission:
  - ICU transfer (already IN the ICU, so we use: transfer to a higher-acuity
    unit or re-intubation)
  - Initiation of mechanical ventilation  (procedureevents)
  - In-hospital death

Usage:
  python src/labels.py --raw_dir data/raw/ \
                       --features data/features.csv \
                       --output data/features_labeled.csv \
                       --window 6
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# itemid for invasive mechanical ventilation in procedureevents
VENT_ITEMIDS = [225792, 225794]   # InvasiveVent, NonInvasiveVent


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_icustays(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "icustays.csv")
    df = pd.read_csv(path, usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                     low_memory=False)
    df["intime"]  = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    return df


def load_admissions(raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, "admissions.csv")
    df = pd.read_csv(path, usecols=["hadm_id", "deathtime", "hospital_expire_flag"],
                     low_memory=False)
    df["deathtime"] = pd.to_datetime(df["deathtime"], errors="coerce")
    return df


def load_procedureevents(raw_dir: str) -> pd.DataFrame:
    """Load ventilation procedure events."""
    path = os.path.join(raw_dir, "procedureevents.csv")
    if not os.path.exists(path):
        log.warning("procedureevents.csv not found — ventilation labels skipped")
        return pd.DataFrame(columns=["stay_id", "starttime", "itemid"])
    df = pd.read_csv(path, usecols=["stay_id", "starttime", "itemid"], low_memory=False)
    df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    df = df[df["itemid"].isin(VENT_ITEMIDS)]
    return df


# ── Label construction ────────────────────────────────────────────────────────

def label_death(icustays: pd.DataFrame,
                admissions: pd.DataFrame,
                window_hours: float) -> pd.DataFrame:
    """
    Label = 1 if patient died within window_hours of ICU admission.
    """
    df = icustays.merge(admissions, on="hadm_id", how="left")
    df["hours_to_death"] = (
        (df["deathtime"] - df["intime"]).dt.total_seconds() / 3600
    )
    df["label_death"] = (
        (df["hospital_expire_flag"] == 1) &
        (df["hours_to_death"] <= window_hours) &
        (df["hours_to_death"] >= 0)
    ).astype(int)
    return df[["stay_id", "label_death"]]


def label_ventilation(icustays: pd.DataFrame,
                      procedureevents: pd.DataFrame,
                      window_hours: float) -> pd.DataFrame:
    """
    Label = 1 if invasive or non-invasive ventilation started within
    window_hours of ICU admission.
    """
    df = procedureevents.merge(
        icustays[["stay_id", "intime"]], on="stay_id", how="left"
    )
    df["hours_to_vent"] = (
        (df["starttime"] - df["intime"]).dt.total_seconds() / 3600
    )
    vent_stays = df[
        (df["hours_to_vent"] > 0) & (df["hours_to_vent"] <= window_hours)
    ]["stay_id"].unique()

    icustays = icustays.copy()
    icustays["label_vent"] = icustays["stay_id"].isin(vent_stays).astype(int)
    return icustays[["stay_id", "label_vent"]]


def combine_labels(icustays: pd.DataFrame,
                   label_death: pd.DataFrame,
                   label_vent: pd.DataFrame) -> pd.DataFrame:
    """
    Combine individual label sources into a single binary 'deteriorated' column.
    Label = 1 if ANY deterioration event occurs within the window.
    """
    df = icustays[["stay_id"]].copy()
    df = df.merge(label_death, on="stay_id", how="left")
    df = df.merge(label_vent,  on="stay_id", how="left")
    df = df.fillna(0)

    df["deteriorated"] = (
        (df["label_death"] == 1) | (df["label_vent"] == 1)
    ).astype(int)

    pos = df["deteriorated"].sum()
    total = len(df)
    log.info("Labels: %d/%d positive (%.1f%%)", pos, total, 100 * pos / total)
    return df[["stay_id", "deteriorated", "label_death", "label_vent"]]


# ── Main ─────────────────────────────────────────────────────────────────────

def run(raw_dir: str,
        features_path: str,
        output_path: str,
        window_hours: float = 6.0) -> pd.DataFrame:

    log.info("Building labels (window = %.0f h)", window_hours)

    icustays       = load_icustays(raw_dir)
    admissions     = load_admissions(raw_dir)
    procedureevents = load_procedureevents(raw_dir)

    lbl_death = label_death(icustays, admissions, window_hours)
    lbl_vent  = label_ventilation(icustays, procedureevents, window_hours)
    labels    = combine_labels(icustays, lbl_death, lbl_vent)

    # Merge labels into the feature snapshot
    features = pd.read_csv(features_path, low_memory=False)
    out = features.merge(labels, on="stay_id", how="inner")

    log.info("Feature+label matrix: %d stays × %d cols", *out.shape)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("Saved → %s", output_path)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",   default="data/raw/")
    parser.add_argument("--features",  default="data/features.csv")
    parser.add_argument("--output",    default="data/features_labeled.csv")
    parser.add_argument("--window",    type=float, default=6.0,
                        help="Hours after ICU admission to look for deterioration")
    args = parser.parse_args()
    run(args.raw_dir, args.features, args.output, args.window)
