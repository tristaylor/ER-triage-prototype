"""
train.py
────────
Trains an XGBoost classifier on the labeled feature matrix and saves the
model artifact + evaluation report.

  - Stratified k-fold cross-validation
  - Class imbalance handling via scale_pos_weight
  - Reports AUROC, AUPRC, sensitivity @ 90% specificity
  - Saves model to models/xgb_model.pkl

Usage:
  python src/train.py --features data/features_labeled.csv \
                      --output models/ \
                      --cv_folds 5
"""

import os
import argparse
import logging
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix, classification_report,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Columns that are IDs / metadata — never fed to the model
DROP_COLS = [
    "stay_id", "subject_id", "hadm_id",
    "label_death", "label_vent",        # component labels
    "deteriorated",                      # target
    "admittime", "dischtime", "intime",  # datetimes if accidentally carried through
]


# ── Feature preparation ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame):
    """Split df into X (features) and y (target), encode any categoricals."""
    y = df["deteriorated"].values

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop)

    # Encode any remaining object columns (should be minimal after features.py)
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    log.info("Feature matrix: %d rows × %d cols | positive rate: %.1f%%",
             *X.shape, 100 * y.mean())
    return X, y


# ── Evaluation helpers ────────────────────────────────────────────────────────

def sensitivity_at_specificity(y_true, y_prob, target_specificity: float = 0.90):
    """Find sensitivity when specificity ≥ target_specificity."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity  = 1 - fpr
    # Find the highest sensitivity where specificity meets the target
    mask = specificity >= target_specificity
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auroc":    round(roc_auc_score(y_true, y_prob), 4),
        "auprc":    round(average_precision_score(y_true, y_prob), 4),
        "sens_at_90spec": round(sensitivity_at_specificity(y_true, y_prob, 0.90), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def get_model(n_pos: int, n_neg: int, **kwargs) -> xgb.XGBClassifier:
    """
    XGBoost with class imbalance correction.
    scale_pos_weight = n_negative / n_positive is the recommended XGBoost
    approach for imbalanced datasets.
    """
    defaults = dict(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return xgb.XGBClassifier(**defaults)


def cross_validate(X: pd.DataFrame, y: np.ndarray,
                   cv_folds: int = 5, **model_kwargs) -> dict:
    """
    Stratified k-fold CV.  Returns fold-level metrics and OOF predictions.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_metrics = []
    oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        n_pos, n_neg = y_tr.sum(), (y_tr == 0).sum()
        model = get_model(n_pos, n_neg, **model_kwargs)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

        metrics = evaluate(y_val, probs)
        fold_metrics.append(metrics)
        log.info("Fold %d | AUROC: %.4f | AUPRC: %.4f | Sens@90spec: %.4f",
                 fold, metrics["auroc"], metrics["auprc"], metrics["sens_at_90spec"])

    # Aggregate OOF metrics
    oof_metrics = evaluate(y, oof_probs)
    log.info("─" * 50)
    log.info("OOF  | AUROC: %.4f | AUPRC: %.4f | Sens@90spec: %.4f",
             oof_metrics["auroc"], oof_metrics["auprc"], oof_metrics["sens_at_90spec"])

    mean_auroc = np.mean([m["auroc"] for m in fold_metrics])
    std_auroc  = np.std([m["auroc"]  for m in fold_metrics])
    log.info("CV   | AUROC mean±std: %.4f ± %.4f", mean_auroc, std_auroc)

    return {
        "fold_metrics":  fold_metrics,
        "oof_metrics":   oof_metrics,
        "mean_auroc":    round(float(mean_auroc), 4),
        "std_auroc":     round(float(std_auroc),  4),
        "oof_probs":     oof_probs.tolist(),
    }


def train_final_model(X: pd.DataFrame, y: np.ndarray, **model_kwargs) -> xgb.XGBClassifier:
    """Train a single model on the full dataset (for deployment)."""
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    model = get_model(n_pos, n_neg, **model_kwargs)
    model.fit(X, y, verbose=False)
    log.info("Final model trained on %d samples", len(y))
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def run(features_path: str,
        output_dir: str,
        cv_folds: int = 5,
        **model_kwargs) -> xgb.XGBClassifier:

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(features_path, low_memory=False)
    X, y = prepare_features(df)

    log.info("Running %d-fold cross-validation …", cv_folds)
    cv_results = cross_validate(X, y, cv_folds=cv_folds, **model_kwargs)

    log.info("Training final model on full dataset …")
    model = train_final_model(X, y, **model_kwargs)

    # Save artefacts
    model_path   = os.path.join(output_dir, "xgb_model.pkl")
    metrics_path = os.path.join(output_dir, "cv_metrics.json")
    features_path_out = os.path.join(output_dir, "feature_names.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in cv_results.items() if k != "oof_probs"}, f, indent=2)
    with open(features_path_out, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    log.info("Saved model    → %s", model_path)
    log.info("Saved metrics  → %s", metrics_path)
    log.info("Saved features → %s", features_path_out)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",    default="data/features_labeled.csv")
    parser.add_argument("--output",      default="models/")
    parser.add_argument("--cv_folds",    type=int,   default=5)
    parser.add_argument("--n_estimators",type=int,   default=200)
    parser.add_argument("--max_depth",   type=int,   default=5)
    parser.add_argument("--learning_rate",type=float,default=0.05)
    args = parser.parse_args()

    model_kwargs = {
        "n_estimators":  args.n_estimators,
        "max_depth":     args.max_depth,
        "learning_rate": args.learning_rate,
    }
    run(args.features, args.output, args.cv_folds, **model_kwargs)
