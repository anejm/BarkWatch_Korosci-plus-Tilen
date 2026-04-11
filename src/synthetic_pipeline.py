"""
synthetic_pipeline.py
---------------------
Orchestrates the synthetic BarkWatch data pipeline:

  1. Preprocess  – feature-engineer synthetic bark beetle timeseries
  2. Export      – split into train_synthetic / val_synthetic / test_synthetic

Mirrors the structure of pipeline.py but operates entirely on synthetic data
produced by generating_synthetic_data/synthetic_pipeline.py.

Usage:
    python synthetic_pipeline.py               # full run, year-based split
    python synthetic_pipeline.py --demo        # small demo dataset (N odseki)
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Split configuration  (matches pipeline.py)
# ---------------------------------------------------------------------------
TRAIN_CUTOFF_YEAR = 2020   # train:  leto  <  TRAIN_CUTOFF_YEAR
VAL_CUTOFF_YEAR   = 2022   # val:    TRAIN_CUTOFF_YEAR <= leto < VAL_CUTOFF_YEAR
                            # test:   leto >= VAL_CUTOFF_YEAR

# Demo mode: keep only this many randomly-sampled odseki (seed=42)
DEMO_N_ODSEKI = 500

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "data_processing"))

import synthetic_processing


# ---------------------------------------------------------------------------
# Step 1 – Preprocessing
# ---------------------------------------------------------------------------

def step_preprocess():
    """
    Call synthetic_processing, save intermediate CSVs, return DataFrames.

    Files written:
      data/synthetic/synthetic_posek_processed.csv  — feature table
      data/synthetic/synthetic_target.csv           — 12-horizon target matrix
    """
    log.info("=== Step 1: Preprocessing synthetic data ===")
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    log.info("  [synthetic] loading and feature-engineering bark beetle data...")
    features_df = synthetic_processing.preprocess()
    features_df.write_csv(SYNTHETIC_DIR / "synthetic_posek_processed.csv")
    log.info(
        f"  [synthetic] {features_df.shape[0]:,} rows × {features_df.shape[1]} cols "
        f"→ synthetic_posek_processed.csv"
    )

    log.info("  [synthetic] building 12-horizon target matrix...")
    target_df = synthetic_processing.make_target()
    target_df.write_csv(SYNTHETIC_DIR / "synthetic_target.csv")
    log.info(
        f"  [synthetic] {target_df.shape[0]:,} rows × {target_df.shape[1]} cols "
        f"→ synthetic_target.csv"
    )

    return features_df, target_df


# ---------------------------------------------------------------------------
# Step 2 – Join features with targets
# ---------------------------------------------------------------------------

def step_join(features_df: pl.DataFrame, target_df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join target horizons onto the feature table on (ggo, odsek, leto_mesec).
    Numeric nulls are filled with 0.
    """
    log.info("=== Step 2: Joining features with targets ===")

    df = features_df.join(target_df, on=["ggo", "odsek", "leto_mesec"], how="left")

    numeric_cols = [
        c for c, t in zip(df.columns, df.dtypes)
        if t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                 pl.UInt32, pl.UInt64, pl.Int8, pl.Int16,
                 pl.UInt8, pl.UInt16)
    ]
    df = df.with_columns([
        pl.col(c).fill_nan(None).fill_null(0) for c in numeric_cols
    ])

    log.info(f"  joined dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ---------------------------------------------------------------------------
# Step 3 – Split and export
# ---------------------------------------------------------------------------

def step_split_export(df: pl.DataFrame, demo: bool = False) -> None:
    """
    Split the dataset by year and export train / val / test CSVs.

    Boundaries (same as pipeline.py):
      train : leto  <  TRAIN_CUTOFF_YEAR
      val   : TRAIN_CUTOFF_YEAR  <= leto  <  VAL_CUTOFF_YEAR
      test  : leto  >= VAL_CUTOFF_YEAR

    Full outputs:
      data/synthetic/splits/train_synthetic.csv
      data/synthetic/splits/val_synthetic.csv
      data/synthetic/splits/test_synthetic.csv
    """
    log.info("=== Step 3: Splitting and exporting ===")

    if "leto" not in df.columns:
        log.info("  'leto' column missing – deriving from 'leto_mesec'")
        df = df.with_columns(
            pl.col("leto_mesec").str.slice(0, 4).cast(pl.Int32).alias("leto")
        )

    log.info(
        f"  split config: train < {TRAIN_CUTOFF_YEAR} | "
        f"val {TRAIN_CUTOFF_YEAR}–{VAL_CUTOFF_YEAR - 1} | "
        f"test >= {VAL_CUTOFF_YEAR}"
    )

    if demo:
        n_available = df.select(["ggo", "odsek"]).unique().shape[0]
        n_sample    = min(DEMO_N_ODSEKI, n_available)
        log.info(f"  [demo] subsampling to {n_sample} of {n_available} [ggo, odsek] pairs (seed=42)...")
        demo_pairs = df.select(["ggo", "odsek"]).unique().sample(n=n_sample, seed=42)
        df = df.join(demo_pairs, on=["ggo", "odsek"], how="inner")

    train = df.filter(pl.col("leto") < TRAIN_CUTOFF_YEAR)
    val   = df.filter(
        (pl.col("leto") >= TRAIN_CUTOFF_YEAR) & (pl.col("leto") < VAL_CUTOFF_YEAR)
    )
    test  = df.filter(pl.col("leto") >= VAL_CUTOFF_YEAR)

    splits_dir = SYNTHETIC_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    tag = "_demo" if demo else ""
    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        split.write_csv(splits_dir / f"{name}_synthetic{tag}.csv")

    log.info(f"  train : {len(train):,} rows → {splits_dir}/train_synthetic{tag}.csv")
    log.info(f"  val   : {len(val):,} rows → {splits_dir}/val_synthetic{tag}.csv")
    log.info(f"  test  : {len(test):,} rows → {splits_dir}/test_synthetic{tag}.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BarkWatch synthetic data pipeline – preprocess and split."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            f"Create small demo datasets ({DEMO_N_ODSEKI} odseki) "
            "suitable for weak computers."
        ),
    )
    args = parser.parse_args()

    log.info("Starting BarkWatch synthetic pipeline%s", " [DEMO MODE]" if args.demo else "")

    features_df, target_df = step_preprocess()
    final_df = step_join(features_df, target_df)
    step_split_export(final_df, demo=args.demo)

    log.info("Synthetic pipeline complete.")


if __name__ == "__main__":
    main()
