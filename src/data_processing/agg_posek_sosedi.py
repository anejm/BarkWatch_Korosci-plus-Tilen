"""
Aggregate odseki data for a given list of odsek IDs.

Designed for use after get_najblizje() from bliznji_odseki.py — takes the
returned list of IDs and produces a single aggregated row representing the
combined forest characteristics of those odseki.

Aggregation strategy (mirrors agg_sestoji_odseke.py):
  - Percentage columns (lzskdv*) + negovan: area-weighted average
  - Absolute columns (lesna zaloga, posek, pomladek): sum

Usage:
    from agg_posek_sosedi import agg_odseki

    # ids is a list of odsek ID strings, e.g. from get_najblizje()
    result = agg_odseki(["31001", "31002", "31003"])
    print(result)
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

# --- Data location ---
_BASE = Path(__file__).parent.parent.parent
ODSEKI_DIR = _BASE / "data" / "processed"

_ODSEKI_FILES = [
    ODSEKI_DIR / "odseki_processed_01.csv",
    ODSEKI_DIR / "odseki_processed_02.csv",
    ODSEKI_DIR / "odseki_processed_03.csv",
    ODSEKI_DIR / "odseki_processed_04.csv",
    ODSEKI_DIR / "odseki_processed_05.csv",
]

# Percentage/ratio columns — aggregated with area-weighted average
WEIGHTED_MEAN_COLS = [
    'lzskdv11', 'lzskdv11_m', 'lzskdv21', 'lzskdv21_m',
    'lzskdv30', 'lzskdv30_m', 'lzskdv34', 'lzskdv34_m',
    'lzskdv39', 'lzskdv39_m', 'lzskdv41', 'lzskdv41_m',
    'lzskdv50', 'lzskdv50_m', 'lzskdv60', 'lzskdv60_m',
    'lzskdv70', 'lzskdv70_m', 'lzskdv80', 'lzskdv80_m',
    'negovan',
]

# Absolute value columns [m3 or ha] — aggregated with sum
SUM_COLS = [
    'pompov', 'lzigl', 'lzlst', 'lzsku',
    'etigl', 'etlst', 'etsku',
]


@lru_cache(maxsize=1)
def _load_all_odseki() -> pd.DataFrame:
    """Load all odseki CSVs into a single DataFrame. Cached after first call."""
    frames = []
    for path in _ODSEKI_FILES:
        if path.exists():
            frames.append(pd.read_csv(path, encoding="utf-8", low_memory=False))
    if not frames:
        raise FileNotFoundError(f"No odseki CSV files found in {ODSEKI_DIR}")
    return pd.concat(frames, ignore_index=True)


def agg_odseki(odsek_ids: list[str]) -> pd.Series:
    """
    Aggregate odseki data for the given list of odsek IDs into a single row.

    Percentage columns are area-weighted averaged; absolute columns are summed.
    The result also includes 'odseki_count' (number of odseki found) and
    'povrsina_skupaj' (total area [ha] of all matched odseki).

    Args:
        odsek_ids: list of odsek ID strings, e.g. as returned by get_najblizje()

    Returns:
        pd.Series with aggregated values.

    Raises:
        ValueError: if none of the given IDs are found in the data
    """
    if not odsek_ids:
        raise ValueError("odsek_ids list is empty.")

    df = _load_all_odseki()

    # odsek column may be numeric in CSV — compare as strings
    df["odsek"] = df["odsek"].astype(str)
    subset = df[df["odsek"].isin([str(i) for i in odsek_ids])].copy()

    if subset.empty:
        raise ValueError(f"None of the given odsek IDs were found: {odsek_ids}")

    weights = pd.to_numeric(subset["povrsina"], errors="coerce").fillna(0)
    total_weight = weights.sum()

    result = {}

    # Area-weighted average for percentage/ratio columns
    for col in WEIGHTED_MEAN_COLS:
        if col not in subset.columns:
            continue
        vals = pd.to_numeric(subset[col], errors="coerce")
        if total_weight > 0:
            result[col] = (vals * weights).sum() / total_weight
        else:
            result[col] = np.nan

    # Sum for absolute columns (m3, ha)
    for col in SUM_COLS:
        if col not in subset.columns:
            continue
        vals = pd.to_numeric(subset[col], errors="coerce")
        result[col] = vals.sum(min_count=1)

    result["odseki_count"] = len(subset)
    result["povrsina_skupaj"] = total_weight

    return pd.Series(result)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agg_posek_sosedi.py <odsek_id1> <odsek_id2> ...")
        sys.exit(1)

    ids = sys.argv[1:]
    print(f"Aggregating {len(ids)} odseki: {ids}")
    print(agg_odseki(ids).to_string())
