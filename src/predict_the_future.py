"""
predict_the_future.py
---------------------
For each (ggo, odsek) in the test set, take the most recent row
(the one with the latest leto_mesec — the last observed data point
for which no actual future values exist yet) and run the trained
sequential models to produce genuinely future h1..h12 predictions.

The h{i}_pred value corresponds to the harvest i months beyond the
base leto_mesec for that odsek.

Inputs:
  models/xgb_models.pkl
  data/processed/splits/test.csv

Output:
  data/predictions/future_predictions.csv
    Columns: ggo, odsek, leto_mesec, h1_pred .. h12_pred
    Values are in m³ (expm1 applied, matching predictions.csv convention).
    leto_mesec is the base month; h{i}_pred is the predicted harvest
    for base_month + i months.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data" / "processed" / "splits"
MODEL_PATH = ROOT / "models" / "xgb_models.pkl"
PRED_DIR   = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH   = PRED_DIR / "future_predictions.csv"

# ---------------------------------------------------------------------------
# Column config  (must match train.py / testing.py)
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


def _two_stage_predict(clf, reg, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    clf_prob = clf.predict_proba(X)[:, 1]
    clf_bin  = (clf_prob >= threshold).astype(int)
    reg_pred = np.maximum(reg.predict(X), 0)
    return clf_bin * reg_pred


def predict(models: dict, X_base: pd.DataFrame) -> np.ndarray:
    """
    Run sequential inference: h_i model receives base features + pred_h1..pred_h(i-1).
    Returns array of shape (n_rows, 12).
    """
    X_aug = X_base.copy()
    preds = np.zeros((len(X_base), 12))

    for i, col in enumerate(TARGET_COLS):
        m            = models[col]
        feature_cols = m["feature_cols"]
        threshold    = m.get("threshold", 0.5)

        # Add any missing prediction columns as zeros
        for fc in feature_cols:
            if fc not in X_aug.columns:
                X_aug[fc] = 0.0

        X_aligned   = X_aug[feature_cols]
        preds[:, i] = _two_stage_predict(m["clf"], m["reg"], X_aligned, threshold)

        if i < 11:
            X_aug = X_aug.copy()
            X_aug[f"pred_{col}"] = preds[:, i]

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading models from {MODEL_PATH} …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    models: dict = joblib.load(MODEL_PATH)
    print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    print(f"\nLoading test set from {test_path.name} …")
    test_x = pd.read_csv(test_path, low_memory=False)
    print(f"  Total test rows: {len(test_x):,}")

    # For each (ggo, odsek), keep only the row with the maximum leto_mesec.
    # This is the last observed data point — the most recent state of each
    # odsek for which no actual future harvest values exist yet.
    group_keys = [c for c in ["ggo", "odsek"] if c in test_x.columns]
    last_rows = (
        test_x
        .sort_values("leto_mesec")
        .groupby(group_keys, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"  Unique (ggo, odsek) pairs selected (last row each): {len(last_rows):,}")
    print(f"  Base months range: {last_rows['leto_mesec'].min()} → {last_rows['leto_mesec'].max()}")

    id_df, X_base = prepare_features(last_rows)
    print(f"  Base features: {X_base.shape[1]}")

    print("\nRunning sequential inference …")
    preds = predict(models, X_base)

    # Convert from log1p space to m³ (matching predictions.csv convention)
    preds_m3 = np.expm1(np.maximum(preds, 0))

    out = pd.concat(
        [id_df, pd.DataFrame(preds_m3, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(OUT_PATH, index=False)
    print(f"\nFuture predictions saved → {OUT_PATH}  ({len(out):,} rows, values in m³)")
    print("  Each h{{i}}_pred corresponds to base leto_mesec + i months.")


if __name__ == "__main__":
    main()
