"""
simulation.py
─────────────
Simulates ER throughput under two conditions:
  1. Baseline   — standard FIFO (first-in, first-out) triage
  2. AI-assisted — queue ordered by deterioration risk score

Metrics evaluated:
  - Median / 90th-pct time-to-treatment for high-risk patients
  - Number of deterioration events that occurred before treatment (misses)
  - Reduction in time-to-treatment vs baseline

Usage:
  python src/simulation.py --model models/xgb_model.pkl \
                           --features data/features_labeled.csv \
                           --output data/simulation_results.json
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Minutes per patient treatment slot — roughly average ER assessment time
TREATMENT_SLOT_MINUTES = 15
# Time window in which a CRITICAL patient "misses" treatment if not seen
DETERIORATION_WINDOW_MINUTES = 120


# ── Queue strategies ──────────────────────────────────────────────────────────

def fifo_queue(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline: order by arrival time (row order = arrival order)."""
    return df.reset_index(drop=True).assign(queue_position=lambda x: x.index + 1)


def ai_queue(df: pd.DataFrame) -> pd.DataFrame:
    """AI-assisted: order by risk score descending."""
    return (df.sort_values("risk_score", ascending=False)
              .reset_index(drop=True)
              .assign(queue_position=lambda x: x.index + 1))


# ── Simulation engine ─────────────────────────────────────────────────────────

def simulate(queue: pd.DataFrame,
             slot_minutes: int = TREATMENT_SLOT_MINUTES,
             deterioration_window: int = DETERIORATION_WINDOW_MINUTES) -> pd.DataFrame:
    """
    Assign a treatment_time_minutes to each patient based on their queue
    position and measure outcomes.
    """
    q = queue.copy()
    q["treatment_time_minutes"] = q["queue_position"] * slot_minutes

    # "Miss" = a CRITICAL/HIGH-risk patient is treated AFTER the deterioration window
    q["is_high_risk"] = q["risk_tier"].isin(["CRITICAL", "HIGH"]).astype(int)
    q["missed"]       = (
        (q["is_high_risk"] == 1) &
        (q["treatment_time_minutes"] > deterioration_window)
    ).astype(int)

    return q


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    high_risk = df[df["is_high_risk"] == 1]
    return {
        "n_patients":             int(len(df)),
        "n_high_risk":            int(len(high_risk)),
        "median_ttt_all_minutes": float(np.median(df["treatment_time_minutes"])),
        "p90_ttt_all_minutes":    float(np.percentile(df["treatment_time_minutes"], 90)),
        "median_ttt_high_risk":   float(np.median(high_risk["treatment_time_minutes"]))
                                  if len(high_risk) else None,
        "p90_ttt_high_risk":      float(np.percentile(high_risk["treatment_time_minutes"], 90))
                                  if len(high_risk) else None,
        "n_missed":               int(df["missed"].sum()),
        "miss_rate_pct":          round(100 * df["missed"].sum() / max(len(high_risk), 1), 2),
    }


def compare(baseline_metrics: dict, ai_metrics: dict) -> dict:
    def _reduction(base, ai):
        if base is None or ai is None:
            return None
        return round(100 * (base - ai) / max(base, 1e-9), 2)

    return {
        "median_ttt_reduction_pct": _reduction(
            baseline_metrics["median_ttt_high_risk"],
            ai_metrics["median_ttt_high_risk"],
        ),
        "p90_ttt_reduction_pct": _reduction(
            baseline_metrics["p90_ttt_high_risk"],
            ai_metrics["p90_ttt_high_risk"],
        ),
        "miss_reduction_absolute": (
            baseline_metrics["n_missed"] - ai_metrics["n_missed"]
        ),
        "miss_reduction_pct": _reduction(
            baseline_metrics["miss_rate_pct"],
            ai_metrics["miss_rate_pct"],
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run(model_path: str,
        features_path: str,
        output_path: str,
        n_patients: int = 200) -> dict:
    """
    End-to-end simulation:
      1. Load features + true labels
      2. Score patients with trained model
      3. Simulate FIFO vs AI-assisted triage
      4. Compare outcomes
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from predict import load_model, load_feature_names, score_dataframe

    model_dir     = os.path.dirname(model_path)
    model         = load_model(model_path)
    feature_names = load_feature_names(model_dir)

    df = pd.read_csv(features_path, low_memory=False)

    # Sample for simulation if dataset is large
    if len(df) > n_patients:
        df = df.sample(n=n_patients, random_state=42).reset_index(drop=True)

    scored = score_dataframe(model, feature_names, df)

    # Carry over ground truth if available
    if "deteriorated" in df.columns:
        scored["deteriorated"] = df["deteriorated"].values

    # Simulate both strategies
    baseline_sim = simulate(fifo_queue(scored))
    ai_sim       = simulate(ai_queue(scored))

    baseline_metrics = compute_metrics(baseline_sim)
    ai_metrics       = compute_metrics(ai_sim)
    comparison       = compare(baseline_metrics, ai_metrics)

    results = {
        "n_patients":        n_patients,
        "baseline":          baseline_metrics,
        "ai_assisted":       ai_metrics,
        "improvement":       comparison,
    }

    # Log summary
    log.info("─" * 55)
    log.info("SIMULATION RESULTS  (n=%d patients)", n_patients)
    log.info("%-35s  %8s  %8s", "", "BASELINE", "AI")
    log.info("%-35s  %8.1f  %8.1f",
             "Median TTT high-risk (min)",
             baseline_metrics["median_ttt_high_risk"] or 0,
             ai_metrics["median_ttt_high_risk"]       or 0)
    log.info("%-35s  %8.1f  %8.1f",
             "90th pct TTT high-risk (min)",
             baseline_metrics["p90_ttt_high_risk"]    or 0,
             ai_metrics["p90_ttt_high_risk"]          or 0)
    log.info("%-35s  %8d  %8d",
             "Missed deteriorations",
             baseline_metrics["n_missed"],
             ai_metrics["n_missed"])
    log.info("─" * 55)
    log.info("Median TTT reduction: %s%%", comparison["median_ttt_reduction_pct"])
    log.info("Misses reduced by   : %d (%.1f%%)",
             comparison["miss_reduction_absolute"],
             comparison["miss_reduction_pct"] or 0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved → %s", output_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="models/xgb_model.pkl")
    parser.add_argument("--features",    default="data/features_labeled.csv")
    parser.add_argument("--output",      default="data/simulation_results.json")
    parser.add_argument("--n_patients",  type=int, default=200)
    args = parser.parse_args()
    run(args.model, args.features, args.output, args.n_patients)
