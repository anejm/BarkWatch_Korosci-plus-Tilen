"""
testing.py
----------
Load saved models and produce predictions on the test set.

Predictions are built sequentially: each horizon h_i uses the base test
features augmented with predictions from models h1..h(i-1), matching the
feature set each model was trained on (stored in model["feature_cols"]).

Inputs:
  models/xgb_models.pkl
  data/test.csv  (or data/test_short.csv with --short)
  data/processed/target.csv  — for evaluation (optional; skipped if absent)

Output:
  data/predictions/predictions.csv
    Columns: <id cols>, h1_pred .. h12_pred
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT / "data" / "processed" / "splits"
MODEL_PATH  = ROOT / "models" / "xgb_models.pkl"
TARGET_PATH = ROOT / "data" / "processed" / "target.csv"
PRED_DIR    = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH   = PRED_DIR / "predictions.csv"

# ---------------------------------------------------------------------------
# Column config  (must match train.py)
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]
DROP_COLS     = [
    "datum", "leto", "target", "log1p_target",
    "sosedi_target_sum", "sosedi_target_mean",
    "sosedi_target_std", "sosedi_target_median",
    "sosedi_log1p_target_mean",
]
TARGET_COLS   = [f"h{h}" for h in range(1, 13)]
PRED_COLS     = [f"h{h}_pred" for h in range(1, 13)]

# When True, targets in target.csv are in log1p space; expm1 before evaluation.
LOG_TARGET: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BarkWatch test-set inference")
    p.add_argument("--short", action="store_true",
                   help="Use test_short.csv instead of test.csv")
    return p.parse_args()


def _join_keys(df_x: pd.DataFrame, df_ref: pd.DataFrame) -> list[str]:
    return sorted(set(df_x.columns) & set(df_ref.columns) & set(POSSIBLE_KEYS))


def load_test(short: bool) -> pd.DataFrame:
    suffix = "_short" if short else ""
    path   = DATA_DIR / f"test{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    print(f"Loading test set from {path.name} …")
    return pd.read_csv(path, low_memory=False)


def prepare_test(test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split test into (id_df, X_base), filling NaN features with 0."""
    id_cols   = [c for c in POSSIBLE_KEYS if c in test_x.columns]
    drop_cols = [c for c in DROP_COLS + TARGET_COLS if c in test_x.columns]
    feat_cols = [c for c in test_x.columns if c not in set(id_cols + drop_cols)]

    id_df = test_x[id_cols].reset_index(drop=True)
    X     = test_x[feat_cols].reset_index(drop=True)

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(models: dict, X_base: pd.DataFrame) -> np.ndarray:
    """
    Run sequential inference: h_i model receives base features + pred_h1..pred_h(i-1).
    Returns array of shape (n_rows, 12).
    """
    X_aug  = X_base.copy()
    preds  = np.zeros((len(X_base), 12))

    for i, col in enumerate(TARGET_COLS):
        m = models[col]
        feature_cols = m["feature_cols"]
        threshold    = m.get("threshold", 0.5)

        # Add any missing prediction columns as zeros
        # (shouldn't happen in normal flow, but guards against edge cases)
        for fc in feature_cols:
            if fc not in X_aug.columns:
                X_aug[fc] = 0.0

        X_aligned    = X_aug[feature_cols]
        preds[:, i]  = _two_stage_predict(m["clf"], m["reg"], X_aligned, threshold)

        # Augment for next horizon
        if i < 11:
            X_aug = X_aug.copy()
            X_aug[f"pred_{col}"] = preds[:, i]

    return preds


# ---------------------------------------------------------------------------
# Evaluation  (runs only when target.csv is available)
# ---------------------------------------------------------------------------

def evaluate(preds: np.ndarray, id_df: pd.DataFrame, test_x: pd.DataFrame) -> None:
    if not TARGET_PATH.exists():
        print("  target.csv not found — skipping evaluation")
        return

    targets = pd.read_csv(TARGET_PATH, low_memory=False)
    keys    = _join_keys(id_df, targets)
    if not keys:
        print("  No join keys between id_df and target.csv — skipping evaluation")
        return

    # Merge id_df with targets
    merged = id_df.merge(targets[keys + TARGET_COLS], on=keys, how="inner")
    if len(merged) == 0:
        print("  No rows matched between predictions and targets — skipping evaluation")
        return

    # Align preds to merged rows via index position (inner merge may reorder)
    # Use a helper merge to get the row correspondence
    id_df_with_pos = id_df.copy()
    id_df_with_pos["_pos"] = np.arange(len(id_df))
    matched = id_df_with_pos.merge(targets[keys + TARGET_COLS], on=keys, how="inner")

    row_idx  = matched["_pos"].values
    preds_ev = preds[row_idx]
    y_true   = matched[TARGET_COLS].values.astype(float)

    if LOG_TARGET:
        y_true   = np.expm1(np.maximum(y_true, 0))
        preds_ev = np.expm1(np.maximum(preds_ev, 0))

    mae  = mean_absolute_error(y_true, preds_ev)
    rmse = root_mean_squared_error(y_true, preds_ev)
    nz   = y_true > 0

    print(f"\nTest metrics ({len(matched):,} matched rows, all 12 horizons):")
    print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}")
    if nz.sum():
        print(f"  Non-zero — MAE: {mean_absolute_error(y_true[nz], preds_ev[nz]):.4f}"
              f"   RMSE: {root_mean_squared_error(y_true[nz], preds_ev[nz]):.4f}")

    print(f"\n  {'':4s}  {'MAE-all':>10s}  {'MAE-nz':>10s}  {'RMSE':>10s}")
    for i, col in enumerate(TARGET_COLS):
        yt     = y_true[:, i]
        yp     = preds_ev[:, i]
        nz_col = yt > 0
        h_mae  = mean_absolute_error(yt, yp)
        h_rmse = root_mean_squared_error(yt, yp)
        h_nz   = mean_absolute_error(yt[nz_col], yp[nz_col]) if nz_col.sum() else float("nan")
        print(f"  {col:4s}  {h_mae:10.4f}  {h_nz:10.4f}  {h_rmse:10.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Loading models from {MODEL_PATH} …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    models: dict = joblib.load(MODEL_PATH)
    print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

    test_x            = load_test(short=args.short)
    id_df, X_base     = prepare_test(test_x)
    print(f"  Test rows: {len(X_base):,}  |  base features: {X_base.shape[1]}")

    print("\nRunning sequential inference …")
    preds = predict(models, X_base)

    evaluate(preds, id_df, test_x)

    # Convert predictions to original m³ space before saving.
    # Models output log1p-transformed values (targets in target.csv are log1p).
    # Saving in m³ makes predictions.csv directly interpretable and consistent
    # with what evaluate() reports (which also applies expm1).
    preds_m3 = np.expm1(np.maximum(preds, 0))
    out = pd.concat(
        [id_df, pd.DataFrame(preds_m3, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved → {PRED_PATH}  ({len(out):,} rows, values in m³)")


if __name__ == "__main__":
    main()
