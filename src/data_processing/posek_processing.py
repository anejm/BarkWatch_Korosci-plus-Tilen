"""
posek_processing.py
-------------------
Produces two artefacts based on posek (harvest) data:

1. Monthly per-odsek time-series feature table (posek_processed.csv)
   Every [ggo, odsek] has an identical number of rows spanning from
   the global min leto_mesec to the global max leto_mesec; months with no
   recorded harvest are filled with kubikov = 0.

2. 36-step ahead target matrix (target.csv)
   For each [ggo, odsek, leto_mesec] the columns h_1 … h_36 hold the
   (optionally log1p-scaled) kubikov values 1–36 months ahead.
   Rows where a future month falls outside the observed data get NaN.

Public API:
    preprocess()          -> pl.DataFrame   # feature table
    make_target_kubikov() -> pl.DataFrame   # 36-horizon target table
    main()                                  # saves both to data/processed/
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
LOG_TARGET: bool = True

MIN_KUBIKOV_TOTAL: float = 200.0   # drop [ggo, odsek] whose total raw kubikov < this

LAGS    = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12]
HORIZON = 12

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
    """
    Load raw posek CSV, aggregate kubikov to monthly sums per odsek, then
    expand each [ggo, odsek] to cover the global date range so every
    group has the same number of rows. Missing months are filled with 0.

    Filtering is applied on raw kubikov sums before any transformation.
    """
    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["ggo"]   = df_raw["ggo"].astype(str).str.strip()
    df_raw["odsek"] = df_raw["odsek"].str.strip()

    df = df_raw[["ggo", "odsek", "kubikov", "posekano"]].copy()
    df = df.rename(columns={"posekano": "datum"})
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["ggo", "odsek", "datum", "kubikov"])

    # Monthly sums per [ggo, odsek]
    df = df.set_index(["ggo", "odsek", "datum"])
    monthly = (
        df.groupby(["ggo", "odsek"])
          .resample("MS", level="datum")["kubikov"]
          .sum()
          .rename("target")
          .reset_index()
    )
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)

    # Filter on raw kubikov totals (before any log transform)
    pair_totals = monthly.groupby(["ggo", "odsek"])["target"].sum()
    valid_pairs = pair_totals[pair_totals >= MIN_KUBIKOV_TOTAL].index
    monthly = monthly.set_index(["ggo", "odsek"]).loc[valid_pairs].reset_index()

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


def _apply_target_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to target in-place if LOG_TARGET is set."""
    if LOG_TARGET:
        df["target"] = np.log1p(df["target"])
    return df


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar columns."""
    df["leto"]       = df["datum"].dt.year
    df["mesec"]      = df["datum"].dt.month
    df["leto_mesec"] = df["datum"].dt.to_period("M").astype(str)
    df["mesec_sin"]  = np.sin(2 * np.pi * df["mesec"] / 12)
    df["mesec_cos"]  = np.cos(2 * np.pi * df["mesec"] / 12)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, diff, and expanding features per [ggo, odsek].

    All features are computed on `target`, which has already been
    log-transformed (if LOG_TARGET) before this function is called.
    """
    grp = df.groupby(["ggo", "odsek"])["target"]

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
    Load, clean, and feature-engineer posek data.

    Every [ggo, odsek] spans the full global date range; months with no
    recorded harvest have target = 0 and all derived features computed from
    that zero-padded series.

    Returns:
        polars DataFrame with monthly per-[ggo, odsek] feature table.
    """
    monthly = _build_monthly()
    monthly = _apply_target_transform(monthly)
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


def make_target_kubikov() -> pl.DataFrame:
    """
    Build the 36-step ahead target matrix.

    For each [ggo, odsek, leto_mesec] the columns h_1 … h_36 contain
    the kubikov (optionally log1p-scaled) values 1–36 months ahead.
    Rows where a future step falls outside the observed range keep NaN —
    no rows are dropped.

    Returns:
        polars DataFrame with columns: ggo, odsek, leto_mesec, h_1 … h_36.
    """
    monthly = _build_monthly()
    monthly = _apply_target_transform(monthly)
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
    target_df = target_df.reset_index(drop=True)
    return pl.from_pandas(target_df)


def main():
    """Run full preprocessing and save outputs to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw posek data …")
    monthly = _build_monthly()
    n_pairs = monthly[["ggo", "odsek"]].drop_duplicates().shape[0]
    n_months = monthly["datum"].nunique()
    print(
        f"  Monthly rows: {len(monthly):,}  |  "
        f"Unique [ggo, odsek]: {n_pairs:,}  |  "
        f"Months per group: {n_months:,}"
    )

    monthly = _apply_target_transform(monthly)
    monthly = _add_calendar(monthly)
    monthly = monthly.sort_values(["ggo", "odsek", "datum"]).reset_index(drop=True)

    print("Computing lag & rolling features per [ggo, odsek] …")
    monthly = _add_lag_features(monthly)

    feature_cols = (
        ["ggo", "odsek", "leto_mesec", "target",
         "mesec_sin", "mesec_cos",
         "diff_1", "expanding_mean"]
        + [f"lag_{l}"          for l in LAGS]
        + [f"rolling_mean_{w}" for w in WINDOWS]
        + [f"rolling_std_{w}"  for w in WINDOWS]
    )
    out_features = monthly[feature_cols].reset_index(drop=True)
    out_features.to_csv(FEATURES_OUT, index=False)
    print(f"  Saved {len(out_features):,} rows  →  {FEATURES_OUT}")

    num_zero = (out_features.target == 0).sum()
    num_non_zero = (out_features.target != 0).sum()
    print(f"Number of 0 targets: {num_zero}, number of non-0 targets: {num_non_zero}")

    print(f"Building target matrix ({HORIZON} horizons) …")
    target_df = make_target_kubikov().to_pandas()
    target_df.to_csv(TARGET_OUT, index=False)
    print(f"  Saved {len(target_df):,} rows  →  {TARGET_OUT}")

    print("\n=== Done ===")
    print(f"  Features : {out_features.shape[1]} columns  ×  {len(out_features):,} rows")
    print(f"  Targets  : {target_df.shape[1] - 3} horizons  ×  {len(target_df):,} rows")


if __name__ == "__main__":
    main()
