"""
extract_current_day_data_synthetic.py
--------------------------------------
From the synthetic test split, extract the single most recent row for every
(ggo, odsek_id) pair — i.e. the row with the globally maximum leto_mesec.

This "current state" snapshot is what predict_the_future_synthetic.py uses as
its input, so it does not need to load the entire test split.

Input:
  data/synthetic/splits/test_synthetic.csv

Output:
  data/synthetic/current_state_synthetic.csv
    One row per (ggo, odsek_id), leto_mesec = the latest month present in the
    test split (determined dynamically).
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "synthetic" / "splits" / "test_synthetic.csv"
OUT_PATH  = ROOT / "data" / "synthetic" / "current_state_synthetic.csv"


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Synthetic test split not found: {TEST_PATH}")

    print(f"Loading synthetic test split from {TEST_PATH} …")
    test = pd.read_csv(TEST_PATH, low_memory=False)
    print(f"  Total rows: {len(test):,}")

    latest_month = test["leto_mesec"].max()
    print(f"  Latest leto_mesec in test split: {latest_month}")

    group_keys = [c for c in ["ggo", "odsek_id"] if c in test.columns]
    current = (
        test
        .sort_values("leto_mesec")
        .groupby(group_keys, sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    print(f"  Unique ({', '.join(group_keys)}) pairs: {len(current):,}")
    print(f"  Base months range: {current['leto_mesec'].min()} → {current['leto_mesec'].max()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(OUT_PATH, index=False)
    print(f"\nCurrent state saved → {OUT_PATH}  ({len(current):,} rows)")


if __name__ == "__main__":
    main()
