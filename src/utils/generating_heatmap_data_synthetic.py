"""
generating_heatmap_data_synthetic.py
--------------------------------------
Produces two flat tables for heatmap visualization of synthetic bark beetle data:

  heatmap_past_data_synthetic.csv
    Historical actual bark beetle counts read directly from bark_beetle_by_odsek.csv.
    ggo is joined from najblizji_odseki_postaje.csv.

  heatmap_future_predictions_synthetic.csv
    Future predictions from future_predictions_synthetic.csv (output of
    predict_the_future_synthetic.py). Each h{i}_pred is expanded to the
    predicted bark beetle count for base_month + i months.

Both files share the schema:
    Columns: ggo, odsek_id, leto_mesec, target, is_a_prediction
    Ordered: ggo, odsek_id, leto_mesec

Inputs:
  data/synthetic/bark_beetle_by_odsek.csv             – actual counts (odsek_id, leto_mesec, bark_beetle_count)
  data/processed/najblizji_odseki_postaje.csv         – odsek_id → ggo mapping
  data/predictions/future_predictions_synthetic.csv   – future predictions (original space)

Outputs:
  data/synthetic/heatmap_past_data_synthetic.csv
  data/synthetic/heatmap_future_predictions_synthetic.csv
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "data" / "synthetic" / "bark_beetle_by_odsek.csv"
POSTAJE_IN  = ROOT / "data" / "processed" / "najblizji_odseki_postaje.csv"
FUTURE_PATH = ROOT / "data" / "predictions" / "future_predictions_synthetic.csv"
OUT_PAST    = ROOT / "data" / "synthetic" / "heatmap_past_data_synthetic.csv"
OUT_FUTURE  = ROOT / "data" / "synthetic" / "heatmap_future_predictions_synthetic.csv"

HORIZON = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_historical_actuals(target_path: Path) -> pd.DataFrame:
    """
    Load actual monthly bark beetle counts from bark_beetle_by_odsek.csv and
    join ggo from najblizji_odseki_postaje.csv.

    Returns DataFrame with columns: ggo, odsek_id, leto_mesec, target, is_a_prediction.
    """
    df = pd.read_csv(target_path, low_memory=False)
    df["odsek_id"]   = df["odsek_id"].astype(str).str.strip()
    df["leto_mesec"] = df["leto_mesec"].astype(str).str.strip()
    df["target"]     = pd.to_numeric(df["bark_beetle_count"], errors="coerce").fillna(0.0)

    # Join ggo from postaje lookup
    postaje = pd.read_csv(POSTAJE_IN, usecols=["ggo", "odsek_id"], low_memory=False)
    postaje["odsek_id"] = postaje["odsek_id"].astype(str).str.strip()
    postaje["ggo"]      = postaje["ggo"].astype(str).str.strip()
    postaje = postaje.drop_duplicates(subset=["odsek_id"])

    df = df.merge(postaje, on="odsek_id", how="left")
    df["ggo"] = df["ggo"].fillna("unknown")

    df["is_a_prediction"] = False
    return df[["ggo", "odsek_id", "leto_mesec", "target", "is_a_prediction"]].reset_index(drop=True)


def expand_future_predictions(future_path: Path) -> pd.DataFrame:
    """
    Expand future_predictions_synthetic.csv into one row per
    (ggo, odsek_id, future_month).

    For each base row (ggo, odsek_id, base_month) and horizon i:
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
    if not TARGET_PATH.exists():
        raise FileNotFoundError(f"Required input not found: {TARGET_PATH}")

    OUT_PAST.parent.mkdir(parents=True, exist_ok=True)

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
    historical.to_csv(OUT_PAST, index=False)
    print(f"Past data saved   → {OUT_PAST}  ({len(historical):,} rows)")

    if not FUTURE_PATH.exists():
        print(f"\nSkipping future predictions — {FUTURE_PATH.name} not found.")
        print("Run predict_the_future_synthetic.py first to generate it.")
        return

    print(f"\nExpanding future predictions from {FUTURE_PATH.name} …")
    future = expand_future_predictions(FUTURE_PATH)
    future = (
        future
        .sort_values(["ggo", "odsek_id", "leto_mesec"])
        .reset_index(drop=True)
        [["ggo", "odsek_id", "leto_mesec", "target", "is_a_prediction"]]
    )
    print(f"  Rows: {len(future):,}  |  "
          f"range: {future['leto_mesec'].min()} → {future['leto_mesec'].max()}")
    future.to_csv(OUT_FUTURE, index=False)
    print(f"Future data saved → {OUT_FUTURE}  ({len(future):,} rows)")


if __name__ == "__main__":
    main()
