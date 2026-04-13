"""
synthetic_pipeline.py
---------------------
Orchestrates the synthetic BarkWatch data pipeline:

  1. Preprocess  – feature-engineer bark_beetle_by_odsek.csv (joins ggo +
                   weather station from najblizji_odseki_postaje, adds lag /
                   rolling / calendar features in the same manner as pipeline.py)
  2. Aggregate   – optionally build enrichment tables (weather, neighbours,
                   odseki+sestoji) — reuses the same agg modules as pipeline.py
  3. Join        – combine selected tables into one feature-rich dataset
  4. Export      – split into train_synthetic / val_synthetic / test_synthetic

Enrichment flags (all off by default):
  --weather   add rolling 12-month weather features per parcel
  --sosedi    add neighbour features per parcel
  --odsek     add static forest-segment (sestoji) features per parcel

Input:
  data/synthetic/bark_beetle_by_odsek.csv   – simulated bark beetle counts per odsek

Usage:
    python synthetic_pipeline.py                              # base features only
    python synthetic_pipeline.py --weather --sosedi --odsek  # fully enriched
    python synthetic_pipeline.py --demo                       # small demo dataset
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

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
PROCESSED_DIR = ROOT / "data" / "processed"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"

# ---------------------------------------------------------------------------
# Split configuration  (matches pipeline.py)
# ---------------------------------------------------------------------------
TRAIN_CUTOFF_YEAR = 2020   # train:  leto  <  TRAIN_CUTOFF_YEAR
VAL_CUTOFF_YEAR   = 2022   # val:    TRAIN_CUTOFF_YEAR <= leto < VAL_CUTOFF_YEAR
                            # test:   leto >= VAL_CUTOFF_YEAR

DEMO_N_ODSEKI = 500

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "data_processing"))

import bark_beetle_processing
import agg_posek_meritve
import agg_posek_sosedi
import agg_sestoji_odseke


# ---------------------------------------------------------------------------
# Step 1 – Preprocessing
# ---------------------------------------------------------------------------

def step_preprocess():
    """
    Call bark_beetle_processing, save intermediate CSVs, return DataFrames.

    Reads bark_beetle_by_odsek.csv, joins ggo and weather station from
    najblizji_odseki_postaje.csv, and adds lag / rolling / calendar features
    in the same manner as posek_processing.py in pipeline.py.

    Files written:
      data/synthetic/bark_beetle_processed.csv  — feature table
                                                   (odsek_id, weather, ggo,
                                                    bark_beetle_count + features)
      data/synthetic/bark_beetle_target.csv     — 12-horizon target matrix
    """
    log.info("=== Step 1: Preprocessing bark beetle data ===")
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    log.info("  [bark_beetle] loading and feature-engineering bark_beetle_by_odsek.csv...")
    features_df = bark_beetle_processing.preprocess()
    features_df.write_csv(SYNTHETIC_DIR / "bark_beetle_processed.csv")
    log.info(
        f"  [bark_beetle] {features_df.shape[0]:,} rows × {features_df.shape[1]} cols "
        f"→ bark_beetle_processed.csv"
    )

    log.info("  [bark_beetle] building 12-horizon target matrix...")
    target_df = bark_beetle_processing.make_target()
    target_df.write_csv(SYNTHETIC_DIR / "bark_beetle_target.csv")
    log.info(
        f"  [bark_beetle] {target_df.shape[0]:,} rows × {target_df.shape[1]} cols "
        f"→ bark_beetle_target.csv"
    )

    return features_df, target_df


# ---------------------------------------------------------------------------
# Step 2 – Aggregate relationships  (conditional on CLI flags)
# ---------------------------------------------------------------------------

def step_aggregate(
    weather: bool,
    sosedi: bool,
    odsek: bool,
) -> dict[str, Optional[pl.DataFrame]]:
    """
    Build the requested enrichment tables and return them in a dict.

    Keys always present (value is None when the flag was not set):
      "meritve"  – rolling 12-month weather features
      "sosedi"   – neighbour posek features
      "sestoji"  – static odsek/segment features
    """
    log.info("=== Step 2: Aggregating relationships ===")

    result: dict[str, Optional[pl.DataFrame]] = {
        "meritve": None,
        "sosedi":  None,
        "sestoji": None,
    }

    if weather:
        log.info("  [agg_posek_meritve] joining posek with rolling weather features...")
        result["meritve"] = agg_posek_meritve.aggregate()
        log.info(
            f"  [agg_posek_meritve] {result['meritve'].shape[0]:,} rows × "
            f"{result['meritve'].shape[1]} cols"
        )
    else:
        log.info("  [agg_posek_meritve] skipped (--weather not set)")

    if sosedi:
        log.info("  [agg_posek_sosedi] joining posek with neighbour posek features...")
        result["sosedi"] = agg_posek_sosedi.aggregate()
        log.info(
            f"  [agg_posek_sosedi] {result['sosedi'].shape[0]:,} rows × "
            f"{result['sosedi'].shape[1]} cols"
        )
    else:
        log.info("  [agg_posek_sosedi] skipped (--sosedi not set)")

    if odsek:
        log.info("  [agg_sestoji_odseke] joining odseki with aggregated sestoji...")
        result["sestoji"] = agg_sestoji_odseke.aggregate()
        log.info(
            f"  [agg_sestoji_odseke] {result['sestoji'].shape[0]:,} rows × "
            f"{result['sestoji'].shape[1]} cols"
        )
    else:
        log.info("  [agg_sestoji_odseke] skipped (--odsek not set)")

    return result


# ---------------------------------------------------------------------------
# Step 3 – Join all tables
# ---------------------------------------------------------------------------

def step_join(
    features_df: pl.DataFrame,
    target_df: pl.DataFrame,
    agg: dict[str, Optional[pl.DataFrame]],
) -> pl.DataFrame:
    """
    Left-join the requested enrichment tables and target onto the bark beetle
    feature table.  Any agg entry that is None is silently skipped.

    The feature table uses odsek_id as the parcel key (matching
    bark_beetle_by_odsek.csv).  Agg tables expose odsek_id so no rename is
    needed before the join.

    Join order (when present):
      features × weather        on (ggo, odsek_id, leto_mesec)
               × neighbours     on (ggo, odsek_id, leto_mesec)
               × odsek/segment  on (ggo, odsek_id)
               × target         on (ggo, odsek_id, leto_mesec)
    """
    log.info("=== Step 3: Joining datasets ===")

    base = features_df
    ggo_dtype = base.schema["ggo"]

    def _align_ggo(df: pl.DataFrame) -> pl.DataFrame:
        if df.schema["ggo"] != ggo_dtype:
            return df.with_columns(pl.col("ggo").cast(ggo_dtype))
        return df

    if agg["meritve"] is not None:
        meritve_clean = _align_ggo(agg["meritve"])
        if "used_station" in meritve_clean.columns:
            meritve_clean = meritve_clean.drop("used_station")
        base = base.join(meritve_clean, on=["ggo", "odsek_id", "leto_mesec"], how="left")
        log.info(f"  after weather join:   {base.shape[0]:,} rows × {base.shape[1]} cols")

    if agg["sosedi"] is not None:
        sosedi_clean = _align_ggo(agg["sosedi"])
        base = base.join(sosedi_clean, on=["ggo", "odsek_id", "leto_mesec"], how="left")
        log.info(f"  after neighbour join: {base.shape[0]:,} rows × {base.shape[1]} cols")

    if agg["sestoji"] is not None:
        sestoji_clean = _align_ggo(agg["sestoji"])
        base = base.join(sestoji_clean, on=["ggo", "odsek_id"], how="left", suffix="_odsek")
        log.info(f"  after odseki join:    {base.shape[0]:,} rows × {base.shape[1]} cols")

    base = base.join(target_df, on=["ggo", "odsek_id", "leto_mesec"], how="left")
    log.info(f"  after target join:    {base.shape[0]:,} rows × {base.shape[1]} cols")

    numeric_cols = [
        c for c, t in zip(base.columns, base.dtypes)
        if t in (pl.Float32, pl.Float64, pl.Int32, pl.Int64,
                 pl.UInt32, pl.UInt64, pl.Int8, pl.Int16,
                 pl.UInt8, pl.UInt16)
    ]
    base = base.with_columns([
        pl.col(c).fill_nan(None).fill_null(0) for c in numeric_cols
    ])

    log.info(f"  final dataset: {base.shape[0]:,} rows × {base.shape[1]} cols")
    return base


# ---------------------------------------------------------------------------
# Step 4 – Split and export
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
    log.info("=== Step 4: Splitting and exporting ===")

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
        n_available = df.select(["ggo", "odsek_id"]).unique().shape[0]
        n_sample    = min(DEMO_N_ODSEKI, n_available)
        log.info(f"  [demo] subsampling to {n_sample} of {n_available} [ggo, odsek_id] pairs (seed=42)...")
        demo_pairs = df.select(["ggo", "odsek_id"]).unique().sample(n=n_sample, seed=42)
        df = df.join(demo_pairs, on=["ggo", "odsek_id"], how="inner")

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
        description="BarkWatch synthetic data pipeline – preprocess, aggregate, split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python synthetic_pipeline.py                              # base features only\n"
            "  python synthetic_pipeline.py --weather --sosedi --odsek  # fully enriched\n"
            "  python synthetic_pipeline.py --sosedi                     # neighbours only\n"
        ),
    )
    parser.add_argument("--demo", action="store_true",
                        help=f"subsample to {DEMO_N_ODSEKI} odseki (weak hardware)")
    parser.add_argument("--weather", action="store_true",
                        help="add rolling 12-month weather features per parcel")
    parser.add_argument("--sosedi", action="store_true",
                        help="add neighbour features from real posek AND synthetic data")
    parser.add_argument("--odsek", action="store_true",
                        help="add static forest-segment (sestoji) features per parcel")
    args = parser.parse_args()

    enrichments = [
        f"{'weather' if args.weather else ''}",
        f"{'sosedi' if args.sosedi else ''}",
        f"{'odsek' if args.odsek else ''}",
    ]
    active = [e for e in enrichments if e]
    label = f" [{', '.join(active)}]" if active else " [base only]"
    log.info("Starting BarkWatch bark-beetle synthetic pipeline%s%s",
             " [DEMO MODE]" if args.demo else "", label)

    features_df, target_df = step_preprocess()
    agg = step_aggregate(weather=args.weather, sosedi=args.sosedi, odsek=args.odsek)
    final_df = step_join(features_df, target_df, agg)
    step_split_export(final_df, demo=args.demo)

    log.info("Synthetic pipeline complete.")


if __name__ == "__main__":
    main()
