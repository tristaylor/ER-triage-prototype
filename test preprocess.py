"""tests/test_preprocess.py"""
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import clean_chartevents, pivot_vitals, ITEMID_MAP


def _mock_chartevents(n=50, seed=0):
    rng = np.random.default_rng(seed)
    item_ids = list(ITEMID_MAP.keys())
    return pd.DataFrame({
        "subject_id": rng.integers(1, 10, n),
        "hadm_id":    rng.integers(100, 110, n),
        "stay_id":    rng.integers(1000, 1010, n),
        "charttime":  pd.date_range("2021-01-01", periods=n, freq="30min"),
        "itemid":     rng.choice(item_ids, n),
        "valuenum":   rng.uniform(60, 120, n),   # HR-range values — not all vitals valid
    })


def test_clean_maps_itemids():
    df = _mock_chartevents()
    clean = clean_chartevents(df)
    assert "vital" in clean.columns
    assert clean["vital"].notna().all()


def test_clean_removes_outliers():
    df = _mock_chartevents()
    # Inject an impossible heart rate
    df.loc[0, "itemid"] = 220045   # heart_rate
    df.loc[0, "valuenum"] = 999    # impossible
    clean = clean_chartevents(df)
    assert (clean[clean["vital"] == "heart_rate"]["valuenum"] <= 300).all()


def test_pivot_creates_wide_format():
    df = _mock_chartevents(n=100)
    clean = clean_chartevents(df)
    wide = pivot_vitals(clean)
    assert "stay_id" in wide.columns
    assert "charttime" in wide.columns
    # At least some vital columns should appear
    vital_cols = [c for c in wide.columns if c not in
                  ["subject_id", "hadm_id", "stay_id", "charttime"]]
    assert len(vital_cols) > 0


def test_no_duplicate_rows_after_pivot():
    df = _mock_chartevents(n=200)
    clean = clean_chartevents(df)
    wide = pivot_vitals(clean)
    dupes = wide.duplicated(subset=["stay_id", "charttime"]).sum()
    assert dupes == 0
