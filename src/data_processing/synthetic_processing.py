"""
synthetic_processing.py
-----------------------
Produces two artefacts from synthetic bark beetle population data:

1. Monthly per-odsek time-series feature table (synthetic_posek_processed.csv)
   Every [ggo, odsek] has an identical number of rows spanning the global date
   range; months not present in the data are filled with bark_beetle_count = 0.

2. 12-step ahead target matrix (synthetic_target.csv)
   For each [ggo, odsek, leto_mesec] the columns h1 … h12 hold the
   (log1p-scaled) bark_beetle_count values 1–12 months ahead.
   Rows where a future month falls outside the observed data get NaN.

Public API:
    preprocess()       -> pl.DataFrame   # feature table
    make_target()      -> pl.DataFrame   # 12-horizon target table
    main()                               # saves both to data/synthetic/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# ---------------------------------------------------------------------------
# Config  (mirrors posek_processing.py)
# ---------------------------------------------------------------------------
LOG_TARGET: bool = True

LAGS    = [1, 3, 6, 12, 24]
WINDOWS = [3, 6, 12]
HORIZON = 12

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
RAW_SYNTH     = SYNTHETIC_DIR / "bark_beetle_population.csv"
FEATURES_OUT  = SYNTHETIC_DIR / "synthetic_posek_processed.csv"
TARGET_OUT    = SYNTHETIC_DIR / "synthetic_target.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_monthly(raw_path: Path = RAW_SYNTH) -> pd.DataFrame:
    """
    Load synthetic bark beetle CSV and expand to a complete monthly grid so
    every [ggo, odsek] has the same number of rows. Missing months → 0.
    """
    df = pd.read_csv(raw_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["ggo"]   = df["ggo"].astype(str).str.strip()
    df["odsek"] = df["odsek"].astype(str).str.strip()
    df = df.dropna(subset=["ggo", "odsek", "leto_mesec", "bark_beetle_count"])
    df["bark_beetle_count"] = pd.to_numeric(df["bark_beetle_count"], errors="coerce").fillna(0.0)

    # Parse leto_mesec as a proper date (first day of month)
    df["datum"] = pd.to_datetime(df["leto_mesec"], format="%Y-%m")

    # Monthly sums per [ggo, odsek] (simulation already monthly, but be safe)
    monthly = (
        df.groupby(["ggo", "odsek", "datum"])["bark_beetle_count"]
          .sum()
          .rename("target")
          .reset_index()
    )
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)

    # Build complete grid: every (ggo, odsek) × every month in global range
    global_min = monthly["datum"].min()
    global_max = monthly["datum"].max()
    all_months = pd.date_range(start=global_min, end=global_max, freq="MS")

    pairs = monthly[["ggo", "odsek"]].drop_duplicates()
    grid  = pairs.assign(key=1).merge(
        pd.DataFrame({"datum": all_months, "key": 1}),
        on="key"
    ).drop(columns="key")

    monthly = grid.merge(monthly, on=["ggo", "odsek", "datum"], how="left")
    monthly["target"] = monthly["target"].fillna(0.0)
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)

    return monthly


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and log-transform columns."""
    df["leto"]         = df["datum"].dt.year
    df["mesec"]        = df["datum"].dt.month
    df["leto_mesec"]   = df["datum"].dt.to_period("M").astype(str)
    df["mesec_sin"]    = np.sin(2 * np.pi * df["mesec"] / 12)
    df["mesec_cos"]    = np.cos(2 * np.pi * df["mesec"] / 12)
    df["log1p_target"] = np.log1p(df["target"])
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, diff, and expanding features per [ggo, odsek]."""
    _base_col = "log1p_target" if LOG_TARGET else "target"
    grp = df.groupby(["ggo", "odsek"])[_base_col]

    for lag in LAGS:
        df[f"lag_{lag}"] = grp.shift(lag)

    for w in WINDOWS:
        shifted = grp.shift(1)
        df[f"rolling_mean_{w}"] = shifted.transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = shifted.transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )

    df["diff_1"]         = grp.diff(1)
    df["expanding_mean"] = grp.shift(1).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess() -> pl.DataFrame:
    """
    Load, clean, and feature-engineer synthetic bark beetle data.

    Every [ggo, odsek] spans the full global date range; months with no
    data have target = 0 and all derived features computed from that
    zero-padded series.

    Returns:
        polars DataFrame with monthly per-[ggo, odsek] feature table.
    """
    monthly = _build_monthly()
    monthly = _add_calendar(monthly)
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)
    monthly = _add_lag_features(monthly)

    feature_cols = (
        ["ggo", "odsek", "leto_mesec",
         "mesec_sin", "mesec_cos",
         "diff_1", "expanding_mean"]
        + [f"lag_{l}"          for l in LAGS]
        + [f"rolling_mean_{w}" for w in WINDOWS]
        + [f"rolling_std_{w}"  for w in WINDOWS]
    )
    out = monthly[feature_cols].reset_index(drop=True)
    return pl.from_pandas(out)


def make_target() -> pl.DataFrame:
    """
    Build the 12-step ahead target matrix from synthetic bark beetle counts.

    For each [ggo, odsek, leto_mesec] the columns h1 … h12 contain the
    bark_beetle_count (optionally log1p-scaled) 1–12 months ahead.
    Rows where a future step falls outside the observed range keep NaN.

    Returns:
        polars DataFrame with columns: ggo, odsek, leto_mesec, h1 … h12.
    """
    monthly = _build_monthly()
    monthly = _add_calendar(monthly)

    base = monthly[["ggo", "odsek", "leto_mesec", "target"]].copy()

    horizon_frames = []
    for h in range(1, HORIZON + 1):
        shifted = (
            base.groupby(["ggo", "odsek"])["target"]
                .shift(-h)
                .rename(f"h{h}")
        )
        horizon_frames.append(shifted)

    target_df = base[["ggo", "odsek", "leto_mesec"]].copy()
    target_df = pd.concat([target_df] + horizon_frames, axis=1)

    if LOG_TARGET:
        h_cols = [f"h{h}" for h in range(1, HORIZON + 1)]
        target_df[h_cols] = np.log1p(target_df[h_cols])

    target_df = target_df.reset_index(drop=True)
    return pl.from_pandas(target_df)


def main():
    """Run full preprocessing and save outputs to data/synthetic/."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading synthetic bark beetle data …")
    monthly = _build_monthly()
    n_pairs  = monthly[["ggo", "odsek"]].drop_duplicates().shape[0]
    n_months = monthly["datum"].nunique()
    print(
        f"  Monthly rows: {len(monthly):,}  |  "
        f"Unique [ggo, odsek]: {n_pairs:,}  |  "
        f"Months per group: {n_months:,}"
    )

    monthly = _add_calendar(monthly)
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)

    print("Computing lag & rolling features per [ggo, odsek] …")
    monthly = _add_lag_features(monthly)

    feature_cols = (
        ["ggo", "odsek", "leto_mesec",
         "mesec_sin", "mesec_cos",
         "diff_1", "expanding_mean"]
        + [f"lag_{l}"          for l in LAGS]
        + [f"rolling_mean_{w}" for w in WINDOWS]
        + [f"rolling_std_{w}"  for w in WINDOWS]
    )
    out_features = monthly[feature_cols].reset_index(drop=True)
    out_features.to_csv(FEATURES_OUT, index=False)
    print(f"  Saved {len(out_features):,} rows  →  {FEATURES_OUT}")

    print(f"Building target matrix ({HORIZON} horizons) …")
    target_df = make_target().to_pandas()
    target_df.to_csv(TARGET_OUT, index=False)
    print(f"  Saved {len(target_df):,} rows  →  {TARGET_OUT}")

    print("\n=== Done ===")
    print(f"  Features : {out_features.shape[1]} columns  ×  {len(out_features):,} rows")
    print(f"  Targets  : {target_df.shape[1] - 3} horizons  ×  {len(target_df):,} rows")


if __name__ == "__main__":
    main()
