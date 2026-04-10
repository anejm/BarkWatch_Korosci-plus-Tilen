"""
posek_processing.py
-------------------
Produces two artefacts based on posek (harvest) data:

1. Monthly per-odsek time-series feature table (posek_processed.csv)
2. 12-step ahead target matrix (target.csv)

Public API:
    preprocess() -> pl.DataFrame   # feature table
    make_target() -> pl.DataFrame  # 12-horizon target table
    main()                         # saves both to data/processed/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# When True: all target-derived features (lags, rolling, diff, expanding) are
# computed in log1p space, and h1..h12 in target.csv are also log1p-scaled.
LOG_TARGET: bool = True

LAGS    = [1, 3, 6, 12, 24]
WINDOWS = [3, 6, 12]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
RAW_POSEK     = ROOT / "data" / "raw" / "ZGS" / "posek.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_OUT  = PROCESSED_DIR / "posek_processed.csv"
TARGET_OUT    = PROCESSED_DIR / "target.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_monthly(raw_path: Path = RAW_POSEK) -> pd.DataFrame:
    """Load raw posek CSV and aggregate kubikov to monthly sums per odsek."""
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["ggo"] = df_raw["ggo"].astype(str).str.strip()
    df_raw["odsek"] = df_raw["odsek"].str.strip()

    df = df_raw[["ggo", "odsek", "kubikov", "posekano"]].copy()
    df = df.rename(columns={"posekano": "datum"})
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["ggo", "odsek", "datum", "kubikov"])

    df = df.set_index(["ggo", "odsek", "datum"])
    monthly = (
        df.groupby(["ggo", "odsek"])
          .resample("MS", level="datum")["kubikov"]
          .sum()
          .rename("target")
          .reset_index()
    )
    return monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and log-transform columns."""
    df["leto"]        = df["datum"].dt.year
    df["mesec"]       = df["datum"].dt.month
    df["leto_mesec"]  = df["datum"].dt.to_period("M").astype(str)
    df["mesec_sin"]   = np.sin(2 * np.pi * df["mesec"] / 12)
    df["mesec_cos"]   = np.cos(2 * np.pi * df["mesec"] / 12)
    df["log1p_target"] = np.log1p(df["target"])
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, diff, and expanding features per odsek."""
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

    df["diff_1"] = grp.diff(1)
    df["expanding_mean"] = grp.shift(1).transform(
        lambda s: s.expanding(min_periods=1).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess() -> pl.DataFrame:
    """
    Load, clean, and feature-engineer posek data.

    Returns:
        polars DataFrame with monthly per-[ggo, odsek] feature table.
    """
    monthly = _build_monthly()
    monthly = _add_calendar(monthly)
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)
    monthly = _add_lag_features(monthly)

    # Excluded from features:
    #   - target / log1p_target: current-month harvest; causes leakage because
    #     train.py can't reliably drop it when it's embedded in the feature CSV.
    #     The lagged features (lag_1, lag_3, …) already encode past activity.
    #   - leto (year): acts as a trend proxy that encourages extrapolation
    #     beyond the training period. Calendar seasonality is captured by
    #     mesec_sin / mesec_cos; long-term trend is captured by expanding_mean.
    #   - datum: raw date string, superseded by leto_mesec.
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
    Build the 12-step ahead target matrix.

    Returns:
        polars DataFrame with columns: ggo, odsek, leto_mesec, h1..h12.
        Only rows with all 12 future months known are included.
    """
    monthly = _build_monthly()
    monthly = _add_calendar(monthly)

    target_pivot = monthly[["ggo", "odsek", "leto_mesec", "target"]].copy()

    horizon_frames = []
    for h in range(1, 13):
        shifted = (
            target_pivot
            .groupby(["ggo", "odsek"])["target"]
            .shift(-h)
            .rename(f"h{h}")
        )
        horizon_frames.append(shifted)

    target_df = target_pivot[["ggo", "odsek", "leto_mesec"]].copy()
    target_df = pd.concat([target_df] + horizon_frames, axis=1)
    target_df = target_df.dropna(subset=[f"h{h}" for h in range(1, 13)])
    target_df = target_df.reset_index(drop=True)

    if LOG_TARGET:
        h_cols = [f"h{h}" for h in range(1, 13)]
        target_df[h_cols] = np.log1p(target_df[h_cols])

    return pl.from_pandas(target_df)


def main():
    """Run full preprocessing and save outputs to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw posek data …")
    monthly = _build_monthly()
    print(
        f"  Monthly rows: {len(monthly):,}  |  "
        f"Unique [ggo, odsek]: {monthly[['ggo', 'odsek']].drop_duplicates().shape[0]:,}"
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

    print("Building target matrix (12 horizons) …")
    target_df = make_target().to_pandas()
    target_df.to_csv(TARGET_OUT, index=False)
    print(f"  Saved {len(target_df):,} rows  →  {TARGET_OUT}")

    print("\n=== Done ===")
    print(f"  Features : {out_features.shape[1]} columns  ×  {len(out_features):,} rows")
    print(f"  Targets  : {target_df.shape[1] - 2} horizons  ×  {len(target_df):,} rows")


if __name__ == "__main__":
    main()
