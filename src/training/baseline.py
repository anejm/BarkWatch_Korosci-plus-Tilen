"""
baseline.py
-----------
Lag-12 seasonal naive baseline.

For every row in the test set, predicts all 12 horizons (h1..h12) using the
lag_12 feature (harvest volume 12 months prior).  Both predictions and targets
are converted from log1p space to original m³ space before evaluation.

Inputs:
  data/processed/splits/test.csv   — test rows with lag_12 and h1..h12
  data/processed/target.csv        — ground-truth targets (optional override)

Output:
  data/predictions/predictions_baseline.csv
    Columns: odsek, leto_mesec, h1_pred .. h12_pred  (values in m³)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
PRED_DIR  = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH = PRED_DIR / "predictions_baseline.csv"

TARGET_COLS = [f"h{h}" for h in range(1, 13)]
PRED_COLS   = [f"h{h}_pred" for h in range(1, 13)]
ID_COLS     = ["ggo", "odsek", "leto_mesec"]


# ---------------------------------------------------------------------------
# Load & predict
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading test set from {TEST_PATH.name} …")
    df = pd.read_csv(TEST_PATH, low_memory=False)
    print(f"  Rows: {len(df):,}  |  lag_12 non-NaN: {df['lag_12'].notna().sum():,}")

    # lag_12 is in log1p space — convert to m³
    lag12_orig = np.expm1(np.maximum(df["lag_12"].fillna(0.0).values, 0.0))

    # Predict all 12 horizons with the same lag-12 value
    preds = np.column_stack([lag12_orig] * 12)

    # Ground-truth targets in log1p space → m³
    y_true = np.expm1(np.maximum(df[TARGET_COLS].values.astype(float), 0.0))

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    valid_mask = ~np.isnan(y_true).any(axis=1)
    if (~valid_mask).sum():
        print(f"  Dropping {(~valid_mask).sum()} rows with NaN targets")
    y_ev = y_true[valid_mask]
    p_ev = preds[valid_mask]

    mae  = mean_absolute_error(y_ev, p_ev)
    rmse = root_mean_squared_error(y_ev, p_ev)
    nz   = y_ev > 0

    print(f"\nBaseline metrics ({len(y_ev):,} rows, all 12 horizons, values in m³):")
    print(f"  Overall  — MAE: {mae:.4f}   RMSE: {rmse:.4f}")
    if nz.sum():
        print(f"  Non-zero — MAE: {mean_absolute_error(y_ev[nz], p_ev[nz]):.4f}"
              f"   RMSE: {root_mean_squared_error(y_ev[nz], p_ev[nz]):.4f}")

    print(f"\n  {'':4s}  {'MAE-all':>10s}  {'MAE-nz':>10s}  {'RMSE':>10s}")
    for i, col in enumerate(TARGET_COLS):
        yt     = y_ev[:, i]
        yp     = p_ev[:, i]
        nz_col = yt > 0
        h_mae  = mean_absolute_error(yt, yp)
        h_rmse = root_mean_squared_error(yt, yp)
        h_nz   = mean_absolute_error(yt[nz_col], yp[nz_col]) if nz_col.sum() else float("nan")
        print(f"  {col:4s}  {h_mae:10.4f}  {h_nz:10.4f}  {h_rmse:10.4f}")

    # ---------------------------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------------------------
    id_cols_present = [c for c in ID_COLS if c in df.columns]
    out = pd.concat(
        [df[id_cols_present].reset_index(drop=True),
         pd.DataFrame(preds, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(PRED_PATH, index=False)
    print(f"\nPredictions saved → {PRED_PATH}  ({len(out):,} rows, values in m³)")


if __name__ == "__main__":
    main()
