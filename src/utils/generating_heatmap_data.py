"""
generating_heatmap_data.py
--------------------------
Combines historical actual harvest data with future model predictions
into a single flat table for heatmap visualization.

Historical actuals are reconstructed from target.csv: for each row
(ggo, odsek, M), h1 = log1p(actual harvxest for M+1 month), so we
shift h1 forward by 1 month and apply expm1 to recover m³ values.

Future predictions come from future_predictions.csv (output of
predict_the_future.py): each h{i}_pred value is the predicted harvest
for base_month + i months.  Each h{i} becomes its own entry in the
output with is_a_prediction=True.

Inputs:
  data/processed/target.csv              – actual h1..h12 (log1p space)
  data/predictions/future_predictions.csv – future predictions (m³)

Output:
  data/processed/heatmap.csv
    Columns: ggo, odsek_id, leto_mesec, target, is_a_prediction
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
TARGET_PATH  = ROOT / "data" / "processed" / "target.csv"
FUTURE_PATH  = ROOT / "data" / "predictions" / "future_predictions.csv"
OUT_PATH     = ROOT / "data" / "processed" / "heatmap.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_historical_actuals(target_path: Path) -> pd.DataFrame:
    """
    Reconstruct actual monthly harvest from target.csv.

    target.csv stores h{i} = log1p(actual harvest for leto_mesec + i months).
    Using h1 (1-step-ahead) for each row: actual for M+1 = expm1(h1[M]).
    This gives one actual observation per (ggo, odsek, leto_mesec) row.

    Returns DataFrame with columns: ggo, odsek_id, leto_mesec, target, is_a_prediction.
    """
    target = pd.read_csv(target_path, low_memory=False)
    if "h1" not in target.columns:
        raise ValueError("target.csv does not contain an 'h1' column")

    target["_period"] = pd.to_datetime(target["leto_mesec"] + "-01")

    # h1[M] = log1p(actual for M+1) → actual month is M+1
    target["actual_leto_mesec"] = (
        (target["_period"] + pd.DateOffset(months=1))
        .dt.to_period("M")
        .astype(str)
    )
    target["target"] = np.expm1(np.maximum(target["h1"], 0))

    actual = target[["ggo", "odsek", "actual_leto_mesec", "target"]].copy()
    actual = actual.rename(columns={"odsek": "odsek_id", "actual_leto_mesec": "leto_mesec"})
    actual["is_a_prediction"] = False

    return actual.reset_index(drop=True)


def expand_future_predictions(future_path: Path) -> pd.DataFrame:
    """
    Expand future_predictions.csv into one row per (ggo, odsek, future_month).

    For each base row (ggo, odsek, base_month):
      h{i}_pred  →  (ggo, odsek, base_month + i months, target=h{i}_pred, is_a_prediction=True)

    Returns DataFrame with columns: ggo, odsek_id, leto_mesec, target, is_a_prediction.
    """
    future = pd.read_csv(future_path, low_memory=False)
    future["_period"] = pd.to_datetime(future["leto_mesec"] + "-01")

    odsek_col = "odsek_id" if "odsek_id" in future.columns else "odsek"

    rows = []
    for h in range(1, 13):
        col = f"h{h}_pred"
        if col not in future.columns:
            continue
        tmp = future[["ggo", odsek_col, "_period", col]].copy()
        tmp["leto_mesec"] = (
            (tmp["_period"] + pd.DateOffset(months=h))
            .dt.to_period("M")
            .astype(str)
        )
        tmp["target"] = tmp[col]
        tmp = tmp.rename(columns={odsek_col: "odsek_id"})
        rows.append(tmp[["ggo", "odsek_id", "leto_mesec", "target"]])

    expanded = pd.concat(rows, ignore_index=True)
    expanded["is_a_prediction"] = True
    return expanded.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    for path in (TARGET_PATH, FUTURE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    print(f"Loading historical actuals from {TARGET_PATH.name} …")
    historical = load_historical_actuals(TARGET_PATH)
    print(f"  Rows: {len(historical):,}  |  "
          f"range: {historical['leto_mesec'].min()} → {historical['leto_mesec'].max()}")

    print(f"\nExpanding future predictions from {FUTURE_PATH.name} …")
    future = expand_future_predictions(FUTURE_PATH)
    print(f"  Rows (12 horizons × {len(future) // 12:,} odseki): {len(future):,}  |  "
          f"range: {future['leto_mesec'].min()} → {future['leto_mesec'].max()}")

    combined = pd.concat([historical, future], ignore_index=True)
    combined = combined.sort_values(["ggo", "odsek_id", "leto_mesec"]).reset_index(drop=True)
    combined = combined[["ggo", "odsek_id", "leto_mesec", "target", "is_a_prediction"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    n_actual = (~combined["is_a_prediction"]).sum()
    n_pred   = combined["is_a_prediction"].sum()
    print(f"\nHeatmap data saved → {OUT_PATH}  ({len(combined):,} rows)")
    print(f"  Historical actuals:  {n_actual:,}  (is_a_prediction=False)")
    print(f"  Future predictions:  {n_pred:,}  (is_a_prediction=True)")
    print(f"  Full date range:     {combined['leto_mesec'].min()} → {combined['leto_mesec'].max()}")


if __name__ == "__main__":
    main()
