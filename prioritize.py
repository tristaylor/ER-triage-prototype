"""
prioritize.py
─────────────
Takes a DataFrame of scored patients and produces a dynamically prioritized
ER queue.

Prioritization logic:
  1. Primary sort: risk_score (descending) — sickest first
  2. Secondary sort: wait_time_minutes (ascending) — tie-break by longest wait
  3. Urgency bump: patients with CRITICAL tier waiting > threshold get moved
     to the front regardless of score (prevents starvation)

Usage (standalone):
  python src/prioritize.py --predictions data/predictions.csv \
                           --output data/queue.csv
"""

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Patients waiting longer than this AND in CRITICAL tier are always top of queue
CRITICAL_WAIT_THRESHOLD_MINUTES = 30


# ── Core prioritization ───────────────────────────────────────────────────────

def prioritize_queue(df: pd.DataFrame,
                     critical_wait_threshold: int = CRITICAL_WAIT_THRESHOLD_MINUTES
                     ) -> pd.DataFrame:
    """
    Rank patients by deterioration risk with starvation prevention.

    Required columns in df:
      - risk_score        : float 0–1
      - risk_tier         : str  CRITICAL / HIGH / MODERATE / LOW
      - wait_time_minutes : float (optional; defaults to 0 if absent)

    Returns a DataFrame sorted by priority with a 'priority_rank' column.
    """
    df = df.copy()

    if "wait_time_minutes" not in df.columns:
        df["wait_time_minutes"] = 0.0

    # Urgency override: CRITICAL patients waiting beyond threshold
    df["urgency_override"] = (
        (df["risk_tier"] == "CRITICAL") &
        (df["wait_time_minutes"] >= critical_wait_threshold)
    ).astype(int)

    # Composite sort: overrides first, then score, then wait time breaks ties
    df = df.sort_values(
        by=["urgency_override", "risk_score", "wait_time_minutes"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    df["priority_rank"] = df.index + 1
    df["priority_label"] = df.apply(_priority_label, axis=1)

    log.info("Queue of %d patients ranked. CRITICAL: %d, HIGH: %d, MODERATE: %d, LOW: %d",
             len(df),
             (df["risk_tier"] == "CRITICAL").sum(),
             (df["risk_tier"] == "HIGH").sum(),
             (df["risk_tier"] == "MODERATE").sum(),
             (df["risk_tier"] == "LOW").sum())
    return df


def _priority_label(row) -> str:
    """Human-readable label for UI display."""
    if row["urgency_override"]:
        return f"⚠ OVERRIDE — waited {int(row['wait_time_minutes'])}min"
    tier_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🟡", "LOW": "🟢"}
    return f"{tier_emoji.get(row['risk_tier'], '')} {row['risk_tier']}"


# ── Simulation helpers ────────────────────────────────────────────────────────

def simulate_arrivals(n_patients: int = 20,
                      seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic ER patient queue for testing / demos.
    Returns a DataFrame with realistic-ish risk scores and wait times.
    """
    rng = np.random.default_rng(seed)

    # Mix of risk tiers roughly matching real ER distributions
    scores = np.concatenate([
        rng.beta(2, 8, int(n_patients * 0.55)),    # LOW    ~55%
        rng.beta(4, 6, int(n_patients * 0.25)),    # MODERATE ~25%
        rng.beta(6, 4, int(n_patients * 0.12)),    # HIGH   ~12%
        rng.beta(8, 2, int(n_patients * 0.08)),    # CRITICAL ~8%
    ])
    scores = scores[:n_patients]
    rng.shuffle(scores)

    wait_times = rng.exponential(scale=45, size=n_patients)  # minutes

    from predict import risk_tier  # local import to avoid circular deps
    df = pd.DataFrame({
        "patient_id":        [f"PT-{i:04d}" for i in range(1, n_patients + 1)],
        "risk_score":        np.round(scores, 4),
        "risk_tier":         [risk_tier(s) for s in scores],
        "wait_time_minutes": np.round(wait_times, 1),
        "arrival_time":      [
            datetime.now().strftime("%H:%M") for _ in range(n_patients)
        ],
    })
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def run(predictions_path: str, output_path: str) -> pd.DataFrame:
    df    = pd.read_csv(predictions_path)
    queue = prioritize_queue(df)
    queue.to_csv(output_path, index=False)
    log.info("Saved queue → %s", output_path)
    return queue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="data/predictions.csv")
    parser.add_argument("--output",      default="data/queue.csv")
    args = parser.parse_args()
    run(args.predictions, args.output)
