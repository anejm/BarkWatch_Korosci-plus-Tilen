"""
test_synthetic.py
-----------------
Load saved synthetic models and produce predictions on the synthetic test set.

No classifier — predictions are built directly from the LGBMRegressor per
horizon.  Sequential augmentation mirrors train_synthetic.py: each h_i model
receives base features plus predictions from h1..h(i-1).

Inputs:
  models/lgb_models_synthetic.pkl
  data/synthetic/splits/test_synthetic.csv    produced by synthetic_pipeline.py
  data/synthetic/bark_beetle_target.csv       produced by bark_beetle_processing

Output:
  data/predictions/predictions_synthetic.csv
    Columns: <id cols>, h1_pred .. h12_pred
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "synthetic" / "splits"
TARGET_PATH = ROOT / "data" / "synthetic" / "bark_beetle_target.csv"
MODEL_PATH  = ROOT / "models" / "lgb_models_synthetic.pkl"
PRED_DIR    = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH   = PRED_DIR / "predictions_synthetic.csv"

sys.path.insert(0, str(ROOT))
from models.model import add_derived_features, sanitize_columns

# ---------------------------------------------------------------------------
# Column config  (must match train_synthetic.py)
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

LOG_TARGET: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_test() -> pd.DataFrame:
    path = DATA_DIR / "test_synthetic.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    print(f"Loading test set from {path.name} …")
    return pd.read_csv(path, low_memory=False)


def prepare_test(test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_cols   = [c for c in POSSIBLE_KEYS if c in test_x.columns]
    drop_cols = [c for c in DROP_COLS + TARGET_COLS if c in test_x.columns]
    feat_cols = [c for c in test_x.columns if c not in set(id_cols + drop_cols)]

    id_df = test_x[id_cols].reset_index(drop=True)
    X     = test_x[feat_cols].reset_index(drop=True)

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling NaN in {len(nan_cols)} columns with 0")
        X = X.fillna(0)

    X = sanitize_columns(X)
    X = add_derived_features(X)
    return id_df, X


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(models: dict, X_base: pd.DataFrame) -> np.ndarray:
    """Sequential inference: h_i model receives base features + pred_h1..pred_h(i-1)."""
    X_aug = X_base.copy()
    preds = np.zeros((len(X_base), 12))

    for i, col in enumerate(TARGET_COLS):
        m            = models[col]
        feature_cols = m["feature_cols"]

        for fc in feature_cols:
            if fc not in X_aug.columns:
                X_aug[fc] = 0.0

        preds[:, i] = np.maximum(m["reg"].predict(X_aug[feature_cols]), 0.0)

        if i < 11:
            X_aug = X_aug.copy()
            X_aug[f"pred_{col}"] = preds[:, i]

    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(preds: np.ndarray, id_df: pd.DataFrame) -> None:
    if not TARGET_PATH.exists():
        print(f"  {TARGET_PATH.name} not found — skipping evaluation")
        return

    targets = pd.read_csv(TARGET_PATH, low_memory=False)
    keys    = sorted(set(id_df.columns) & set(targets.columns) & set(POSSIBLE_KEYS))
    if not keys:
        print("  No join keys between id_df and target.csv — skipping evaluation")
        return

    id_df_with_pos = id_df.copy()
    id_df_with_pos["_pos"] = np.arange(len(id_df))
    matched = id_df_with_pos.merge(targets[keys + TARGET_COLS], on=keys, how="inner")

    if len(matched) == 0:
        print("  No rows matched between predictions and targets — skipping evaluation")
        return

    row_idx  = matched["_pos"].values
    preds_ev = preds[row_idx]
    y_true   = matched[TARGET_COLS].values.astype(float)

    valid_mask = ~np.isnan(y_true).any(axis=1)
    if valid_mask.sum() < len(y_true):
        print(f"  Dropping {(~valid_mask).sum()} rows with NaN targets")
    y_true   = y_true[valid_mask]
    preds_ev = preds_ev[valid_mask]

    if LOG_TARGET:
        y_true   = np.expm1(np.maximum(y_true, 0))
        preds_ev = np.expm1(np.maximum(preds_ev, 0))

    mae  = mean_absolute_error(y_true, preds_ev)
    rmse = root_mean_squared_error(y_true, preds_ev)

    print(f"\nTest metrics ({len(y_true):,} matched rows, all 12 horizons):")
    print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}")

    print(f"\n  {'':4s}  {'MAE':>10s}  {'RMSE':>10s}")
    for i, col in enumerate(TARGET_COLS):
        yt    = y_true[:, i]
        yp    = preds_ev[:, i]
        print(f"  {col:4s}  {mean_absolute_error(yt, yp):10.4f}  {root_mean_squared_error(yt, yp):10.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading models from {MODEL_PATH} …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    models: dict = joblib.load(MODEL_PATH)
    print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

    test_x        = load_test()
    id_df, X_base = prepare_test(test_x)
    print(f"  Test rows: {len(X_base):,}  |  base features: {X_base.shape[1]}")

    print("\nRunning sequential inference …")
    preds = predict(models, X_base)

    evaluate(preds, id_df)

    preds_orig = np.expm1(np.maximum(preds, 0))
    out = pd.concat(
        [id_df, pd.DataFrame(preds_orig, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved → {PRED_PATH}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
