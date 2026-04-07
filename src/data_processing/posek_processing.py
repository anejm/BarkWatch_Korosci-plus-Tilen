"""
posek_processing.py
-------------------
Produces two artefacts based on the EDA in notebooks/posek_timeseries_analysis.ipynb:

1. data/processed/posek_processed.csv
   Monthly per-odsek time-series feature table (whole dataset, no split).
   Includes lag, rolling, and seasonal features derived from ACF/PACF and
   STL decomposition analysis.

2. data/processed/target.csv
   For every (odsek, leto_mesec) row that has 12 complete future months,
   columns h1..h12 hold the raw kubikov target values at each horizon.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# When True: all target-derived features (lags, rolling, diff, expanding) are
# computed in log1p space, and h1..h12 in target.csv are also log1p-scaled.
# testing.py must set the same flag so predictions are expm1'd before export.
LOG_TARGET: bool = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_POSEK   = ROOT / "data" / "raw" / "ZGS" / "posek.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_OUT = PROCESSED_DIR / "posek_processed.csv"
TARGET_OUT   = PROCESSED_DIR / "target.csv"

# ---------------------------------------------------------------------------
# 1. Load & monthly aggregation  (mirrors notebook cell 3)
# ---------------------------------------------------------------------------
print("Loading raw posek data …")
df_raw = pd.read_csv(RAW_POSEK, low_memory=False)
df_raw.columns = df_raw.columns.str.strip()
df_raw["odsek"] = df_raw["odsek"].str.strip()

df = df_raw[["odsek", "kubikov", "posekano"]].copy()
df = df.rename(columns={"posekano": "datum"})
df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
df = df.dropna(subset=["odsek", "datum", "kubikov"])

# Monthly sum of kubikov per odsek; resample fills every month in range
df = df.set_index(["odsek", "datum"])
monthly = (
    df.groupby("odsek")
      .resample("MS", level="datum")["kubikov"]
      .sum()
      .rename("target")
      .reset_index()
)
monthly = monthly.sort_values(["odsek", "datum"]).reset_index(drop=True)
print(f"  Monthly rows: {len(monthly):,}  |  Odseki: {monthly['odsek'].nunique():,}")

# ---------------------------------------------------------------------------
# 2. Calendar features
# ---------------------------------------------------------------------------
monthly["leto"]      = monthly["datum"].dt.year
monthly["mesec"]     = monthly["datum"].dt.month
monthly["leto_mesec"] = monthly["datum"].dt.to_period("M").astype(str)

# Cyclical month encoding (from seasonality analysis — strong 12-month cycle)
monthly["mesec_sin"] = np.sin(2 * np.pi * monthly["mesec"] / 12)
monthly["mesec_cos"] = np.cos(2 * np.pi * monthly["mesec"] / 12)

# ---------------------------------------------------------------------------
# 3. Log-transform  (notebook: skewed right, log1p stabilises variance)
# ---------------------------------------------------------------------------
monthly["log1p_target"] = np.log1p(monthly["target"])

# ---------------------------------------------------------------------------
# 4. Per-odsek lag & rolling features
#    Lags chosen from ACF/PACF: strong signal at 1, 3, 6, 12, 24
# ---------------------------------------------------------------------------
LAGS    = [1, 3, 6, 12, 24]
WINDOWS = [3, 6, 12]

print("Computing lag & rolling features per odsek …")

# Sort to ensure correct ordering for per-odsek time-series operations
monthly = monthly.sort_values(["odsek", "datum"]).reset_index(drop=True)

# Use log1p_target as the base series when LOG_TARGET is enabled so that
# all derived features live in the same space as the targets the model learns.
_base_col = "log1p_target" if LOG_TARGET else "target"
grp = monthly.groupby("odsek")[_base_col]

for lag in LAGS:
    monthly[f"lag_{lag}"] = grp.shift(lag)

for w in WINDOWS:
    shifted = grp.shift(1)
    monthly[f"rolling_mean_{w}"] = shifted.transform(
        lambda s: s.rolling(w, min_periods=1).mean()
    )
    monthly[f"rolling_std_{w}"] = shifted.transform(
        lambda s: s.rolling(w, min_periods=1).std()
    )

monthly["diff_1"] = grp.diff(1)

monthly["expanding_mean"] = grp.shift(1).transform(
    lambda s: s.expanding(min_periods=1).mean()
)

# ---------------------------------------------------------------------------
# 5. Save posek_processed.csv
# ---------------------------------------------------------------------------
feature_cols = (
    ["odsek", "datum", "leto_mesec", "leto",
     "mesec_sin", "mesec_cos",
     "target", "log1p_target", "diff_1",
     "expanding_mean"]
    + [f"lag_{l}"          for l in LAGS]
    + [f"rolling_mean_{w}" for w in WINDOWS]
    + [f"rolling_std_{w}"  for w in WINDOWS]
)

out_features = monthly[feature_cols].reset_index(drop=True)
out_features.to_csv(FEATURES_OUT, index=False)
print(f"  Saved {len(out_features):,} rows  →  {FEATURES_OUT}")

# ---------------------------------------------------------------------------
# 6. Build target.csv  (12-step ahead targets per odsek × month)
#    Row index: (odsek, leto_mesec)
#    Columns:   h1 … h12  (future raw kubikov values)
# ---------------------------------------------------------------------------
print("Building target matrix (12 horizons) …")

target_pivot = monthly[["odsek", "leto_mesec", "target"]].copy()

# For each horizon h, shift the target back by h within each odsek
horizon_frames = []
for h in range(1, 13):
    shifted = (
        target_pivot
        .groupby("odsek")["target"]
        .shift(-h)
        .rename(f"h{h}")
    )
    horizon_frames.append(shifted)

target_df = target_pivot[["odsek", "leto_mesec"]].copy()
target_df = pd.concat([target_df] + horizon_frames, axis=1)

# Keep only rows where ALL 12 future months are known
target_df = target_df.dropna(subset=[f"h{h}" for h in range(1, 13)])
target_df = target_df.reset_index(drop=True)

if LOG_TARGET:
    h_cols = [f"h{h}" for h in range(1, 13)]
    target_df[h_cols] = np.log1p(target_df[h_cols])

target_df.to_csv(TARGET_OUT, index=False)
print(f"  Saved {len(target_df):,} rows  →  {TARGET_OUT}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Done ===")
print(f"  Features : {out_features.shape[1]} columns  ×  {len(out_features):,} rows")
print(f"  Targets  : {target_df.shape[1] - 2} horizons  ×  {len(target_df):,} rows")
print(f"  Output dir: {PROCESSED_DIR}")
