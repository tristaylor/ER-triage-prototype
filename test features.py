"""tests/test_features.py"""
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import (
    add_derived_vitals, add_flag_features,
    add_delta_features, impute_and_clip, build_admission_snapshot,
)


def _mock_vitals(n=100, seed=1):
    rng = np.random.default_rng(seed)
    n_stays = 10
    stay_ids = np.repeat(np.arange(n_stays), n // n_stays)

    return pd.DataFrame({
        "subject_id":           stay_ids,
        "hadm_id":              stay_ids + 100,
        "stay_id":              stay_ids,
        "charttime":            pd.date_range("2021-01-01", periods=n, freq="1h"),
        "heart_rate":           rng.uniform(60,  120, n),
        "sbp":                  rng.uniform(90,  160, n),
        "dbp":                  rng.uniform(60,  100, n),
        "resp_rate":            rng.uniform(12,  25,  n),
        "spo2":                 rng.uniform(92,  100, n),
        "temp_c":               rng.uniform(36,  38.5, n),
        "anchor_age":           rng.integers(25, 80, n),
        "sex_male":             rng.integers(0, 2, n),
        "hospital_expire_flag": np.zeros(n, dtype=int),
        "admission_type":       rng.choice(["EMERGENCY", "ELECTIVE"], n),
        "admittime":            pd.date_range("2021-01-01", periods=n, freq="1h"),
        "intime":               pd.date_range("2021-01-01", periods=n, freq="1h"),
        "hours_since_icu_admit": rng.uniform(0, 6, n),
    })


def test_shock_index_computed():
    df = add_derived_vitals(_mock_vitals())
    assert "shock_index" in df.columns
    assert df["shock_index"].notna().all()


def test_shock_index_values():
    df = _mock_vitals(n=10)
    df["heart_rate"] = 100
    df["sbp"] = 100
    out = add_derived_vitals(df)
    assert (out["shock_index"] == 1.0).all()


def test_flags_binary():
    df = add_flag_features(add_derived_vitals(_mock_vitals()))
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    for col in flag_cols:
        assert df[col].isin([0, 1]).all(), f"{col} is not binary"


def test_low_spo2_flag():
    df = _mock_vitals(n=5)
    df["spo2"] = [90, 95, 93, 97, 91]   # 90, 93, 91 should be flagged
    out = add_flag_features(add_derived_vitals(df))
    expected = [1, 0, 1, 0, 1]
    assert list(out["flag_low_spo2"]) == expected


def test_delta_features_created():
    df = add_delta_features(_mock_vitals())
    assert "heart_rate_delta" in df.columns


def test_impute_fills_nans():
    df = _mock_vitals()
    df.loc[0, "heart_rate"] = np.nan
    df.loc[5, "spo2"]       = np.nan
    out = impute_and_clip(df)
    assert out["heart_rate"].isna().sum() == 0
    assert out["spo2"].isna().sum() == 0


def test_snapshot_one_row_per_stay():
    df = add_flag_features(add_derived_vitals(_mock_vitals()))
    snapshot = build_admission_snapshot(df, first_n_hours=6.0)
    assert snapshot["stay_id"].nunique() == snapshot.shape[0]
