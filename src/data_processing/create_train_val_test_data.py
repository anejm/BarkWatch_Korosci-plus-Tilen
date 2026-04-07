"""
create_train_val_test_data.py
-----------------------------
Joins posek_processed.csv (features) with target.csv (12-horizon targets)
on (odsek, leto_mesec) and produces a temporal train / val / test split.

Split boundaries (inclusive):
  train : leto_mesec  <  VAL_START
  val   : VAL_START  <=  leto_mesec  <  TEST_START
  test  :              leto_mesec  >= TEST_START

Outputs (data/processed/splits/):
  train.csv
  val.csv
  test.csv
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR    = PROCESSED_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = PROCESSED_DIR / "posek_processed.csv"
TARGET_PATH   = PROCESSED_DIR / "target.csv"

# ---------------------------------------------------------------------------
# Split boundaries  (Period strings, compared lexicographically)
# ---------------------------------------------------------------------------
VAL_START  = "2022-01"   # first month of validation set
TEST_START = "2023-07"   # first month of test set

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading features …")
features = pd.read_csv(FEATURES_PATH, low_memory=False)

print("Loading targets …")
targets = pd.read_csv(TARGET_PATH)

# ---------------------------------------------------------------------------
# 2. Set index
# ---------------------------------------------------------------------------
INDEX = ["odsek", "leto_mesec"]

features = features.set_index(INDEX)
targets  = targets.set_index(INDEX)

# ---------------------------------------------------------------------------
# 3. Join  (inner: keep only rows that have both features and all 12 targets)
# ---------------------------------------------------------------------------
df = features.join(targets, how="inner")
print(f"Joined dataset: {len(df):,} rows  |  {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 4. Temporal split on leto_mesec (second level of the MultiIndex)
# ---------------------------------------------------------------------------
leto_mesec = df.index.get_level_values("leto_mesec")

mask_train = leto_mesec < VAL_START
mask_val   = (leto_mesec >= VAL_START) & (leto_mesec < TEST_START)
mask_test  = leto_mesec >= TEST_START

train = df[mask_train]
val   = df[mask_val]
test  = df[mask_test]

print(f"\nSplit summary:")
print(f"  train : {len(train):>7,} rows  ({train.index.get_level_values('leto_mesec').min()} – {train.index.get_level_values('leto_mesec').max()})")
print(f"  val   : {len(val):>7,} rows  ({val.index.get_level_values('leto_mesec').min()} – {val.index.get_level_values('leto_mesec').max()})")
print(f"  test  : {len(test):>7,} rows  ({test.index.get_level_values('leto_mesec').min()} – {test.index.get_level_values('leto_mesec').max()})")

# ---------------------------------------------------------------------------
# 5. Save  (index=True preserves odsek + leto_mesec columns)
# ---------------------------------------------------------------------------
train_path = SPLITS_DIR / "train.csv"
val_path   = SPLITS_DIR / "val.csv"
test_path  = SPLITS_DIR / "test.csv"

train.to_csv(train_path)
val.to_csv(val_path)
test.to_csv(test_path)

print(f"\nSaved splits to {SPLITS_DIR}/")
print(f"  {train_path.name}")
print(f"  {val_path.name}")
print(f"  {test_path.name}")
