"""
generating_heatmap_data.py
--------------------------
Produces two flat tables for heatmap visualization:

  heatmap_past_data.csv
    Historical actual harvest reconstructed from target.csv.
    For each row (ggo, odsek, M) and horizon h_i, the actual harvest for
    month M+i is expm1(h_i). All 12 horizons are unpivoted and deduplicated
    so each (ggo, odsek_id, leto_mesec) appears exactly once.

  heatmap_future_predictions.csv
    Future predictions from future_predictions.csv (output of
    predict_the_future.py). Each h{i}_pred is expanded to the predicted
    harvest for base_month + i months.

Both files share the schema:
    Columns: ggo, odsek_id, leto_mesec, target, is_a_prediction
    Ordered: ggo, odsek_id, leto_mesec

Inputs:
  data/processed/target.csv               – actual h1..h12 (log1p space)
  data/predictions/future_predictions.csv – future predictions (m³)

Outputs:
  data/processed/heatmap_past_data.csv
  data/processed/heatmap_future_predictions.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
TARGET_PATH   = ROOT / "data" / "processed" / "target.csv"
FUTURE_PATH   = ROOT / "data" / "predictions" / "future_predictions.csv"
OUT_PAST      = ROOT / "data" / "processed" / "heatmap_past_data.csv"
OUT_FUTURE    = ROOT / "data" / "processed" / "heatmap_future_predictions.csv"

HORIZON = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_historical_actuals(target_path: Path) -> pd.DataFrame:
    """
    Reconstruct actual monthly harvest from target.csv.

    For each row (ggo, odsek, M) and each horizon i in 1..12:
      h{i}[M] = log1p(actual harvest for M+i months)
    All horizons are unpivoted into individual rows, then deduplicated
    so each (ggo, odsek_id, leto_mesec) keeps the observation from the
    smallest horizon (most recent base month = least forecasting error).

    Returns DataFrame with columns: ggo, odsek_id, leto_mesec, target, is_a_prediction.
    """
    target = pd.read_csv(target_path, low_memory=False)

    h_cols = [f"h{i}" for i in range(1, HORIZON + 1)]
    missing = [c for c in h_cols if c not in target.columns]
    if missing:
        raise ValueError(f"target.csv is missing columns: {missing}")

    odsek_col = "odsek_id" if "odsek_id" in target.columns else "odsek"
    target["ggo"]      = target["ggo"].astype(str)
    target[odsek_col]  = target[odsek_col].astype(str)
    target["_period"]  = pd.to_datetime(target["leto_mesec"] + "-01")

    rows = []
    for i, col in enumerate(h_cols, start=1):
        tmp = target[["ggo", odsek_col, "_period", col]].copy()
        tmp["leto_mesec"] = (
            (tmp["_period"] + pd.DateOffset(months=i))
            .dt.to_period("M")
            .astype(str)
        )
        tmp["target"]   = np.expm1(np.maximum(tmp[col], 0))
        tmp["_horizon"] = i
        rows.append(tmp[["ggo", odsek_col, "leto_mesec", "target", "_horizon"]])

    unpivoted = pd.concat(rows, ignore_index=True)
    unpivoted = unpivoted.rename(columns={odsek_col: "odsek_id"})

    # Keep the observation from the smallest horizon (most direct reading)
    unpivoted = (
        unpivoted
        .sort_values(["ggo", "odsek_id", "leto_mesec", "_horizon"])
        .drop_duplicates(subset=["ggo", "odsek_id", "leto_mesec"], keep="first")
        .drop(columns="_horizon")
    )

    unpivoted["is_a_prediction"] = False
    return unpivoted.reset_index(drop=True)


def expand_future_predictions(future_path: Path) -> pd.DataFrame:
    """
    Expand future_predictions.csv into one row per (ggo, odsek_id, future_month).

    For each base row (ggo, odsek, base_month) and horizon i:
      h{i}_pred → (ggo, odsek_id, base_month + i months, target=h{i}_pred)

    Returns DataFrame with columns: ggo, odsek_id, leto_mesec, target, is_a_prediction.
    """
    future = pd.read_csv(future_path, low_memory=False)

    odsek_col = "odsek_id" if "odsek_id" in future.columns else "odsek"
    future["ggo"]     = future["ggo"].astype(str)
    future[odsek_col] = future[odsek_col].astype(str)
    future["_period"] = pd.to_datetime(future["leto_mesec"] + "-01")

    rows = []
    for i in range(1, HORIZON + 1):
        col = f"h{i}_pred"
        if col not in future.columns:
            continue
        tmp = future[["ggo", odsek_col, "_period", col]].copy()
        tmp["leto_mesec"] = (
            (tmp["_period"] + pd.DateOffset(months=i))
            .dt.to_period("M")
            .astype(str)
        )
        tmp["target"] = tmp[col].where(tmp[col] >= 1, 0)
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
    historical = (
        historical
        .sort_values(["ggo", "odsek_id", "leto_mesec"])
        .reset_index(drop=True)
        [["ggo", "odsek_id", "leto_mesec", "target", "is_a_prediction"]]
    )
    print(f"  Rows: {len(historical):,}  |  "
          f"range: {historical['leto_mesec'].min()} → {historical['leto_mesec'].max()}")

    print(f"Expanding future predictions from {FUTURE_PATH.name} …")
    future = expand_future_predictions(FUTURE_PATH)
    future = (
        future
        .sort_values(["ggo", "odsek_id", "leto_mesec"])
        .reset_index(drop=True)
        [["ggo", "odsek_id", "leto_mesec", "target", "is_a_prediction"]]
    )
    print(f"  Rows: {len(future):,}  |  "
          f"range: {future['leto_mesec'].min()} → {future['leto_mesec'].max()}")

    OUT_PAST.parent.mkdir(parents=True, exist_ok=True)

    historical.to_csv(OUT_PAST, index=False)
    print(f"\nPast data saved   → {OUT_PAST}  ({len(historical):,} rows)")

    future.to_csv(OUT_FUTURE, index=False)
    print(f"Future data saved → {OUT_FUTURE}  ({len(future):,} rows)")


if __name__ == "__main__":
    main()
