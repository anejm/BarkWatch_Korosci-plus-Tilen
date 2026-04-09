"""
testing.py
----------
Loads the two-stage models (XGBClassifier + XGBRegressor per horizon) and
produces predictions on test.csv.

Each horizon's prediction = clf.predict(X) * max(reg.predict(X), 0)
  → 0 when the classifier says no event, regressor magnitude otherwise.

Input:
  models/xgb_models.pkl
  data/processed/splits/test.csv

Output:
  data/predictions/predictions.csv
    Columns: ggo, odsek, leto_mesec, h1_pred .. h12_pred
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "xgb_models.pkl"
TEST_PATH  = ROOT / "data" / "processed" / "splits" / "test.csv"
PRED_DIR   = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH  = PRED_DIR / "predictions.csv"

# ---------------------------------------------------------------------------
# Config  (must match posek_processing.py)
# ---------------------------------------------------------------------------
# When True, target.csv h1..h12 are in log1p space and predictions must be
# expm1'd back to raw kubikov before evaluation and export.
LOG_TARGET: bool = True

# ---------------------------------------------------------------------------
# Columns  (must match train.py)
# ---------------------------------------------------------------------------
INDEX_COLS  = ["ggo", "odsek", "leto_mesec"]
DROP_COLS   = ["datum"]
TARGET_COLS = [f"h{h}" for h in range(1, 13)]
PRED_COLS   = [f"h{h}_pred" for h in range(1, 13)]


def _resolve_test_path() -> Path:
  """Prefer top-level data/test.csv (dummy workflow), fallback to processed split."""
  test_top = ROOT / "data" / "test.csv"
  return test_top if test_top.exists() else TEST_PATH


def _present_index_cols(df: pd.DataFrame) -> list[str]:
  cols = [c for c in INDEX_COLS if c in df.columns]
  required = {"odsek", "leto_mesec"}
  if not required.issubset(set(cols)):
    missing = ", ".join(sorted(required - set(cols)))
    raise ValueError(f"Missing required identifier columns: {missing}")
  return cols

# ---------------------------------------------------------------------------
# 1. Load models
# ---------------------------------------------------------------------------
print(f"Loading models from {MODEL_PATH} …")
models: dict = joblib.load(MODEL_PATH)
print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

# ---------------------------------------------------------------------------
# 2. Load test set
# ---------------------------------------------------------------------------
print("Loading test set …")
test_path = _resolve_test_path()
test = pd.read_csv(test_path)
present_index_cols = _present_index_cols(test)

index   = test[present_index_cols].copy()
X_test  = test.drop(columns=DROP_COLS + present_index_cols + TARGET_COLS, errors="ignore")
y_test  = test[TARGET_COLS].copy()

# Drop rows whose lag/rolling features are NaN (first rows per odsek)
valid_mask   = X_test.notna().all(axis=1)
X_clean      = X_test[valid_mask].reset_index(drop=True)
index_clean  = index[valid_mask].reset_index(drop=True)
y_clean      = y_test[valid_mask].reset_index(drop=True)

dropped = (~valid_mask).sum()
print(f"  Rows after dropna: {len(X_clean):,}  (dropped {dropped:,} with NaN features)")

# ---------------------------------------------------------------------------
# 3. Predict
# ---------------------------------------------------------------------------
preds = np.zeros((len(X_clean), len(TARGET_COLS)))

for i, col in enumerate(TARGET_COLS):
    stage = models[col]
    clf_pred = stage["clf"].predict(X_clean)               # 0 or 1
    reg_pred = np.maximum(stage["reg"].predict(X_clean), 0)
    preds[:, i] = clf_pred * reg_pred

# ---------------------------------------------------------------------------
# 3b. Invert log transform so metrics and exports are in raw kubikov space
# ---------------------------------------------------------------------------
if LOG_TARGET:
    preds   = np.expm1(np.maximum(preds, 0))
    y_clean = pd.DataFrame(np.expm1(y_clean.values), columns=y_clean.columns)

# ---------------------------------------------------------------------------
# 4. Evaluate against ground-truth targets
# ---------------------------------------------------------------------------
mae  = mean_absolute_error(y_clean, preds)
rmse = root_mean_squared_error(y_clean, preds)

y_clean_arr = y_clean.values
nz = y_clean_arr > 0

print(f"\nTest metrics (averaged over all 12 horizons):")
print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}")
if nz.sum():
    mae_nz  = mean_absolute_error(y_clean_arr[nz], preds[nz])
    rmse_nz = root_mean_squared_error(y_clean_arr[nz], preds[nz])
    print(f"  Non-zero — MAE: {mae_nz:.4f}   RMSE: {rmse_nz:.4f}")

print(f"\n  {'':4s}  {'MAE-all':>10s}  {'MAE-nz':>10s}  {'RMSE':>10s}")
for i, col in enumerate(TARGET_COLS):
    y_true_col = y_clean_arr[:, i]
    y_pred_col = preds[:, i]
    h_mae  = mean_absolute_error(y_true_col, y_pred_col)
    h_rmse = root_mean_squared_error(y_true_col, y_pred_col)
    nz_col = y_true_col > 0
    h_mae_nz = mean_absolute_error(y_true_col[nz_col], y_pred_col[nz_col]) if nz_col.sum() else float("nan")
    print(f"  {col:4s}  {h_mae:10.4f}  {h_mae_nz:10.4f}  {h_rmse:10.4f}")

# ---------------------------------------------------------------------------
# 5. Save predictions
# ---------------------------------------------------------------------------
out = pd.concat([index_clean, pd.DataFrame(preds, columns=PRED_COLS)], axis=1)
out.to_csv(PRED_PATH, index=False)
print(f"\nPredictions saved → {PRED_PATH}  ({len(out):,} rows)")
