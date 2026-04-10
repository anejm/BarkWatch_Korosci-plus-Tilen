"""
extract_current_day_data.py
---------------------------
From the full test split, extract the single most recent row for every
(ggo, odsek) pair — i.e. the row with the globally maximum leto_mesec.

This "current state" snapshot is what predict_the_future.py uses as its
input, so it does not need to load the entire test split.

Input:
  data/processed/splits/test.csv

Output:
  data/processed/current_state.csv
    One row per (ggo, odsek), leto_mesec = the latest month present in the
    test split (determined dynamically).
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
TEST_PATH  = ROOT / "data" / "processed" / "splits" / "test.csv"
OUT_PATH   = ROOT / "data" / "processed" / "current_state.csv"


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_PATH}")

    print(f"Loading test split from {TEST_PATH} …")
    test = pd.read_csv(TEST_PATH, low_memory=False)
    print(f"  Total rows: {len(test):,}")

    latest_month = test["leto_mesec"].max()
    print(f"  Latest leto_mesec in test split: {latest_month}")

    group_keys = [c for c in ["ggo", "odsek"] if c in test.columns]
    current = (
        test
        .sort_values("leto_mesec")
        .groupby(group_keys, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"  Unique (ggo, odsek) pairs: {len(current):,}")
    print(f"  Base months range: {current['leto_mesec'].min()} → {current['leto_mesec'].max()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_PATH, index=False)
    print(f"\nCurrent state saved → {OUT_PATH}  ({len(current):,} rows)")


if __name__ == "__main__":
    main()
