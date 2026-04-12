"""
testing.py
----------
Load saved models and produce predictions on the test set.

Predictions are built sequentially: each horizon h_i uses the base test
features augmented with predictions from models h1..h(i-1), matching the
feature set each model was trained on (stored in model["feature_cols"]).

Inputs:
  models/lgb_models.pkl
  data/processed/splits/test.csv  (or test_short.csv with --short)
  data/processed/target.csv  — for evaluation (optional; skipped if absent)

Output:
  data/predictions/predictions.csv
    Columns: <id cols>, h1_pred .. h12_pred

For synthetic data inference, use test_synthetic.py instead.
"""

import argparse
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
DATA_DIR    = ROOT / "data" / "processed" / "splits"
MODEL_PATH  = ROOT / "models" / "lgb_models.pkl"
TARGET_PATH = ROOT / "data" / "processed" / "target.csv"
PRED_DIR    = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH   = PRED_DIR / "predictions.csv"

SYNTHETIC_DATA_DIR    = ROOT / "data" / "synthetic" / "splits"
SYNTHETIC_TARGET_PATH = ROOT / "data" / "synthetic" / "synthetic_target.csv"
MODEL_PATH_SYNTHETIC  = ROOT / "models" / "lgb_models_synthetic.pkl"
PRED_PATH_SYNTHETIC   = PRED_DIR / "predictions_synthetic.csv"

# Allow importing feature-engineering helpers from project root
sys.path.insert(0, str(ROOT))
from models.model import add_derived_features, sanitize_columns

# ---------------------------------------------------------------------------
# Column config  (must match train.py)
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]
DROP_COLS     = [
    "datum", "leto",
]
TARGET_COLS   = [f"h{h}" for h in range(1, 13)]
PRED_COLS     = [f"h{h}_pred" for h in range(1, 13)]

# When True, targets in target.csv are in log1p space; expm1 before evaluation.
LOG_TARGET: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BarkWatch test-set inference")
    p.add_argument("--short", action="store_true",
                   help="Use test_short.csv instead of test.csv")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic test split and lgb_models_synthetic.pkl")
    return p.parse_args()


def _join_keys(df_x: pd.DataFrame, df_ref: pd.DataFrame) -> list[str]:
    return sorted(set(df_x.columns) & set(df_ref.columns) & set(POSSIBLE_KEYS))


def load_test(short: bool, synthetic: bool = False) -> pd.DataFrame:
    if synthetic:
        path = SYNTHETIC_DATA_DIR / "test_synthetic.csv"
    else:
        suffix = "_short" if short else ""
        path   = DATA_DIR / f"test{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {path}")
    print(f"Loading test set from {path.name} …")
    return pd.read_csv(path, low_memory=False)


def prepare_test(test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split test into (id_df, X_base), filling NaN features with 0.

    Applies the same sanitise → derive pipeline used in training so that
    the feature set matches what each model was trained on.
    """
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

def evaluate(
    preds: np.ndarray,
    id_df: pd.DataFrame,
    test_x: pd.DataFrame,
    target_path: Path = TARGET_PATH,
) -> None:
    if not target_path.exists():
        print(f"  {target_path.name} not found — skipping evaluation")
        return

    targets = pd.read_csv(target_path, low_memory=False)
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

    # Drop rows where any target horizon is NaN
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
    nz   = y_true > 0

    print(f"\nTest metrics ({len(y_true):,} matched rows, all 12 horizons):")
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

    model_path  = MODEL_PATH_SYNTHETIC  if args.synthetic else MODEL_PATH
    target_path = SYNTHETIC_TARGET_PATH if args.synthetic else TARGET_PATH
    pred_path   = PRED_PATH_SYNTHETIC   if args.synthetic else PRED_PATH

    print(f"Loading models from {model_path} …")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    models: dict = joblib.load(model_path)
    print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

    test_x        = load_test(short=args.short, synthetic=args.synthetic)
    id_df, X_base = prepare_test(test_x)
    print(f"  Test rows: {len(X_base):,}  |  base features: {X_base.shape[1]}")

    print("\nRunning sequential inference …")
    preds = predict(models, X_base)

    evaluate(preds, id_df, test_x, target_path=target_path)

    # Convert predictions to original space (log1p → expm1) before saving.
    preds_m3 = np.expm1(np.maximum(preds, 0))
    out = pd.concat(
        [id_df, pd.DataFrame(preds_m3, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(pred_path, index=False)
    label = " (synthetic)" if args.synthetic else " (values in m³)"
    print(f"\nPredictions saved → {pred_path}  ({len(out):,} rows{label})")


if __name__ == "__main__":
    main()
