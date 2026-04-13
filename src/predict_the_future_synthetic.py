"""
predict_the_future_synthetic.py
--------------------------------
Loads the synthetic current-state snapshot produced by
extract_current_day_data_synthetic.py (one row per (ggo, odsek_id), the most
recent observed month) and runs the trained single-stage regressor models to
produce genuinely future h1..h12 predictions.

The synthetic models contain only a regressor (no classifier stage), so
prediction is a direct regressor call per horizon.

The h{i}_pred value corresponds to the bark beetle count i months beyond the
base leto_mesec for that odsek_id.

Inputs:
  models/lgb_models_synthetic.pkl
  data/synthetic/current_state_synthetic.csv  ← run extract_current_day_data_synthetic.py first

Output:
  data/predictions/future_predictions_synthetic.csv
    Columns: ggo, odsek_id, leto_mesec, h1_pred .. h12_pred
    Values are in original bark beetle count space (expm1 applied).
    leto_mesec is the base month; h{i}_pred is the predicted count
    for base_month + i months.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
CURRENT_PATH = ROOT / "data" / "synthetic" / "current_state_synthetic.csv"
MODEL_PATH   = ROOT / "models" / "lgb_models_synthetic.pkl"
PRED_DIR     = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = PRED_DIR / "future_predictions_synthetic.csv"

# ---------------------------------------------------------------------------
# Column config  (must match train_synthetic.py / test_synthetic.py)
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]
DROP_COLS     = [
    "datum", "leto", "target", "log1p_target",
    "sosedi_target_sum", "sosedi_target_mean",
    "sosedi_target_std", "sosedi_target_median",
    "sosedi_log1p_target_mean",
]
TARGET_COLS = [f"h{h}" for h in range(1, 13)]
PRED_COLS   = [f"h{h}_pred" for h in range(1, 13)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (id_df, X_base), filling NaN features with 0."""
    id_cols   = [c for c in POSSIBLE_KEYS if c in df.columns]
    drop_cols = [c for c in DROP_COLS + TARGET_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in set(id_cols + drop_cols)]

    id_df = df[id_cols].reset_index(drop=True)
    X     = df[feat_cols].reset_index(drop=True)

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling NaN in {len(nan_cols)} columns with 0")
        X = X.fillna(0)

    return id_df, X


def predict(models: dict, X_base: pd.DataFrame) -> np.ndarray:
    """
    Run sequential inference using single-stage regressors.
    Each h_i model receives base features + pred_h1..pred_h(i-1).
    Returns array of shape (n_rows, 12).
    """
    X_aug = X_base.copy()
    preds = np.zeros((len(X_base), 12))

    for i, col in enumerate(TARGET_COLS):
        m            = models[col]
        feature_cols = m["feature_cols"]

        # Add any missing prediction columns as zeros
        for fc in feature_cols:
            if fc not in X_aug.columns:
                X_aug[fc] = 0.0

        preds[:, i] = np.maximum(m["reg"].predict(X_aug[feature_cols]), 0.0)

        if i < 11:
            X_aug = X_aug.copy()
            X_aug[f"pred_{col}"] = preds[:, i]

    return preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate future bark beetle predictions from the synthetic model."
    )
    parser.add_argument(
        "--h1",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Multiply the h1 prediction by this factor (0.0–5.0) before feeding "
            "it into subsequent models. Default: 1.0 (no change)."
        ),
    )
    args = parser.parse_args()
    if not (0.0 <= args.h1 <= 5.0):
        parser.error(f"--h1 must be between 0.0 and 5.0, got {args.h1}")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Loading synthetic models from {MODEL_PATH} …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    models: dict = joblib.load(MODEL_PATH)
    print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(
            f"Current state file not found: {CURRENT_PATH}\n"
            "Run extract_current_day_data_synthetic.py first."
        )
    print(f"\nLoading current state from {CURRENT_PATH.name} …")
    current = pd.read_csv(CURRENT_PATH, low_memory=False)
    print(f"  Rows: {len(current):,}")
    print(f"  Base months range: {current['leto_mesec'].min()} → {current['leto_mesec'].max()}")

    id_df, X_base = prepare_features(current)
    print(f"  Base features: {X_base.shape[1]}")

    if args.h1 != 1.0:
        print(f"\nH1 multiplier: ×{args.h1}")

    print("\nRunning sequential inference …")
    preds = predict(models, X_base)

    # Apply h1 multiplier after inference if requested
    if args.h1 != 1.0:
        preds[:, 0] = preds[:, 0] * args.h1

    # Convert from log1p space to original bark beetle count space
    preds_orig = np.expm1(np.maximum(preds, 0))

    out = pd.concat(
        [id_df, pd.DataFrame(preds_orig, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(OUT_PATH, index=False)
    print(f"\nFuture predictions saved → {OUT_PATH}  ({len(out):,} rows)")
    print("  Each h{i}_pred corresponds to base leto_mesec + i months (bark beetles / m²).")


if __name__ == "__main__":
    main()
