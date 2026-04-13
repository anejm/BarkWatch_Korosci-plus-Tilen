"""
bark_beetle_processing.py
--------------------------
Produces two artefacts from bark_beetle_by_odsek.csv:

1. Monthly per-odsek time-series feature table (bark_beetle_processed.csv)
   Enriched with ggo and weather station from najblizji_odseki_postaje.csv.
   Every [ggo, odsek_id] has an identical number of rows spanning the global
   date range; months not present in the data are filled with target = 0.

2. 12-step ahead target matrix (bark_beetle_target.csv)
   For each [ggo, odsek_id, leto_mesec] the columns h1 … h12 hold the
   (log1p-scaled) bark_beetle_count values 1–12 months ahead.
   Rows where a future month falls outside the observed data get NaN.

Columns in output feature table:
  ggo, odsek_id, weather, leto_mesec, bark_beetle_count,
  mesec_sin, mesec_cos, diff_1, expanding_mean,
  lag_1, lag_2, lag_3, lag_6, lag_12, lag_24,
  rolling_mean_3, rolling_mean_6, rolling_mean_12,
  rolling_std_3, rolling_std_6, rolling_std_12

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

LAGS    = [1, 2, 3, 6, 12, 24]
WINDOWS = [3, 6, 12]
HORIZON = 12

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_SYNTH    = SYNTHETIC_DIR / "bark_beetle_by_odsek.csv"
POSTAJE_IN   = PROCESSED_DIR / "najblizji_odseki_postaje.csv"
FEATURES_OUT = SYNTHETIC_DIR / "bark_beetle_processed.csv"
TARGET_OUT   = SYNTHETIC_DIR / "bark_beetle_target.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_postaje_lookup() -> pd.DataFrame:
    """
    Load najblizji_odseki_postaje.csv and return a deduplicated mapping of
    odsek_id → ggo, weather (preferred station_id: station_23 > station_123).
    """
    postaje = pd.read_csv(POSTAJE_IN, low_memory=False)
    postaje["odsek_id"] = postaje["odsek_id"].astype(str).str.strip()
    postaje["ggo"]      = postaje["ggo"].astype(str).str.strip()

    def _pick_station(row: pd.Series) -> float:
        s23  = row.get("station_23")
        s123 = row.get("station_123")
        if pd.notna(s23):
            return float(s23)
        if pd.notna(s123):
            return float(s123)
        return np.nan

    postaje["weather"] = postaje.apply(_pick_station, axis=1)
    return postaje[["odsek_id", "ggo", "weather"]].drop_duplicates(subset=["odsek_id"])


def _build_monthly(raw_path: Path = RAW_SYNTH) -> pd.DataFrame:
    """
    Load bark_beetle_by_odsek.csv, join ggo and weather from postaje lookup,
    then expand to a complete monthly grid so every [ggo, odsek_id] has the
    same number of rows. Missing months are filled with target = 0.
    """
    df = pd.read_csv(raw_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df["odsek_id"] = df["odsek_id"].astype(str).str.strip()
    df = df.dropna(subset=["odsek_id", "leto_mesec", "bark_beetle_count"])
    df["bark_beetle_count"] = pd.to_numeric(df["bark_beetle_count"], errors="coerce").fillna(0.0)

    # Join ggo and weather station from postaje lookup
    postaje = _load_postaje_lookup()
    df = df.merge(postaje, on="odsek_id", how="left")
    df["ggo"]     = df["ggo"].fillna("unknown")
    df["weather"] = df["weather"].fillna(-1).astype(int)

    df["datum"] = pd.to_datetime(df["leto_mesec"], format="%Y-%m")

    # Monthly sums per [ggo, odsek_id] (data is already monthly, but aggregate safely)
    monthly = (
        df.groupby(["ggo", "odsek_id", "datum", "weather"])["bark_beetle_count"]
          .sum()
          .rename("target")
          .reset_index()
    )
    monthly = monthly.sort_values(["ggo", "odsek_id", "datum"]).reset_index(drop=True)

    # Build complete grid: every (ggo, odsek_id) × every month in global range
    global_min = monthly["datum"].min()
    global_max = monthly["datum"].max()
    all_months = pd.date_range(start=global_min, end=global_max, freq="MS")

    # Keep static weather per odsek_id (first occurrence)
    odsek_meta = monthly[["ggo", "odsek_id", "weather"]].drop_duplicates(subset=["odsek_id"])

    grid = odsek_meta.assign(key=1).merge(
        pd.DataFrame({"datum": all_months, "key": 1}),
        on="key"
    ).drop(columns="key")

    monthly = grid.merge(
        monthly.drop(columns="weather"),
        on=["ggo", "odsek_id", "datum"],
        how="left",
    )
    monthly["target"] = monthly["target"].fillna(0.0)
    monthly = monthly.sort_values(["ggo", "odsek_id", "datum"]).reset_index(drop=True)

    return monthly


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar columns and log-transformed target."""
    df["leto"]         = df["datum"].dt.year
    df["mesec"]        = df["datum"].dt.month
    df["leto_mesec"]   = df["datum"].dt.to_period("M").astype(str)
    df["mesec_sin"]    = np.sin(2 * np.pi * df["mesec"] / 12)
    df["mesec_cos"]    = np.cos(2 * np.pi * df["mesec"] / 12)
    df["log1p_target"] = np.log1p(df["target"])
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag, rolling, diff, and expanding features per [ggo, odsek_id].
    Features are computed on log1p_target when LOG_TARGET is True.
    """
    base_col = "log1p_target" if LOG_TARGET else "target"
    grp = df.groupby(["ggo", "odsek_id"])[base_col]

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
    Load, clean, and feature-engineer bark_beetle_by_odsek data.

    Joins ggo and weather station from najblizji_odseki_postaje.csv.
    Every [ggo, odsek_id] spans the full global date range; months with no
    data have target = 0 and all derived features computed from that
    zero-padded series.

    Returns:
        polars DataFrame with monthly per-[ggo, odsek_id] feature table.
    """
    monthly = _build_monthly()
    monthly = _add_calendar(monthly)
    monthly = monthly.sort_values(["ggo", "odsek_id", "datum"]).reset_index(drop=True)
    monthly = _add_lag_features(monthly)

    feature_cols = (
        ["ggo", "odsek_id", "weather", "leto_mesec",
         "bark_beetle_count",
         "mesec_sin", "mesec_cos",
         "diff_1", "expanding_mean"]
        + [f"lag_{l}"          for l in LAGS]
        + [f"rolling_mean_{w}" for w in WINDOWS]
        + [f"rolling_std_{w}"  for w in WINDOWS]
    )
    # bark_beetle_count in output = raw (unlogged) count for interpretability
    monthly["bark_beetle_count"] = monthly["target"]
    out = monthly[feature_cols].reset_index(drop=True)
    return pl.from_pandas(out)


def make_target() -> pl.DataFrame:
    """
    Build the 12-step ahead target matrix from bark beetle counts.

    For each [ggo, odsek_id, leto_mesec] the columns h1 … h12 contain the
    bark_beetle_count (optionally log1p-scaled) 1–12 months ahead.
    Rows where a future step falls outside the observed range keep NaN.

    Returns:
        polars DataFrame with columns: ggo, odsek_id, leto_mesec, h1 … h12.
    """
    monthly = _build_monthly()
    monthly = _add_calendar(monthly)

    base = monthly[["ggo", "odsek_id", "leto_mesec", "target"]].copy()

    horizon_frames = []
    for h in range(1, HORIZON + 1):
        shifted = (
            base.groupby(["ggo", "odsek_id"])["target"]
                .shift(-h)
                .rename(f"h{h}")
        )
        horizon_frames.append(shifted)

    target_df = base[["ggo", "odsek_id", "leto_mesec"]].copy()
    target_df = pd.concat([target_df] + horizon_frames, axis=1)

    if LOG_TARGET:
        h_cols = [f"h{h}" for h in range(1, HORIZON + 1)]
        target_df[h_cols] = np.log1p(target_df[h_cols])

    target_df = target_df.reset_index(drop=True)
    return pl.from_pandas(target_df)


def main():
    """Run full preprocessing and save outputs to data/synthetic/."""
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading bark_beetle_by_odsek data …")
    monthly = _build_monthly()
    n_pairs  = monthly[["ggo", "odsek_id"]].drop_duplicates().shape[0]
    n_months = monthly["datum"].nunique()
    print(
        f"  Monthly rows: {len(monthly):,}  |  "
        f"Unique [ggo, odsek_id]: {n_pairs:,}  |  "
        f"Months per group: {n_months:,}"
    )

    monthly = _add_calendar(monthly)
    monthly = monthly.sort_values(["ggo", "odsek_id", "datum"]).reset_index(drop=True)

    print("Computing lag & rolling features per [ggo, odsek_id] …")
    monthly = _add_lag_features(monthly)

    feature_cols = (
        ["ggo", "odsek_id", "weather", "leto_mesec",
         "bark_beetle_count",
         "mesec_sin", "mesec_cos",
         "diff_1", "expanding_mean"]
        + [f"lag_{l}"          for l in LAGS]
        + [f"rolling_mean_{w}" for w in WINDOWS]
        + [f"rolling_std_{w}"  for w in WINDOWS]
    )
    monthly["bark_beetle_count"] = monthly["target"]
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
    print(f"\nSample (first 5 rows):")
    print(out_features.head().to_string(index=False))


if __name__ == "__main__":
    main()
