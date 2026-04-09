"""
pipeline.py
-----------
Orchestrates the full BarkWatch data pipeline:

  1. Preprocess  – clean raw sources (posek, odseki, sestoji, meritve)
  2. Aggregate   – build relationship tables (weather join, neighbour join,
                   odseki+sestoji join)
  3. Join        – combine all tables into one feature-rich dataset
  4. Export      – split into train / val / test and write CSVs

Usage:
    python pipeline.py               # full run, year-based split
    python pipeline.py --demo        # small demo dataset (N odseki)
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
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR    = ROOT / "data"

# ---------------------------------------------------------------------------
# Split configuration  (edit these to change the temporal boundaries)
# ---------------------------------------------------------------------------
TRAIN_CUTOFF_YEAR = 2020   # train:  leto  <  TRAIN_CUTOFF_YEAR
VAL_CUTOFF_YEAR   = 2022   # val:    TRAIN_CUTOFF_YEAR <= leto < VAL_CUTOFF_YEAR
                            # test:   leto >= VAL_CUTOFF_YEAR

# Demo mode: keep only this many randomly-sampled odseki (seed=42)
DEMO_N_ODSEKI = 500

# ---------------------------------------------------------------------------
# Module imports – add data_processing to path so modules resolve cleanly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent / "data_processing"))

import posek_processing
import odseki_processing
import sestoji_processing
import meritve_processing
import agg_posek_meritve
import agg_posek_sosedi
import agg_sestoji_odseke
import najblizji_odseki_postaje_predracun


# ---------------------------------------------------------------------------
# Step 1 – Preprocessing
# ---------------------------------------------------------------------------

def step_preprocess():
    """
    Call each preprocessing module, save intermediate CSVs, and return
    the resulting polars DataFrames.

    Intermediate files written:
      data/processed/posek_processed.csv
      data/processed/odseki_processed.csv
      data/processed/sestoji_processed.csv
      data/processed/vreme_mesecno.csv
    """
    log.info("=== Step 1: Preprocessing ===")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("  [posek] loading and feature-engineering harvest data...")
    posek_df = posek_processing.preprocess()
    posek_df.write_csv(PROCESSED_DIR / "posek_processed.csv")
    log.info(f"  [posek] {posek_df.shape[0]:,} rows × {posek_df.shape[1]} cols "
             f"→ posek_processed.csv")

    log.info("  [odseki] loading forest compartment GeoPackage...")
    odseki_df = odseki_processing.preprocess()
    odseki_df.write_csv(PROCESSED_DIR / "odseki_processed.csv")
    log.info(f"  [odseki] {odseki_df.shape[0]:,} rows × {odseki_df.shape[1]} cols "
             f"→ odseki_processed.csv")

    log.info("  [sestoji] loading forest stand GeoPackage...")
    sestoji_df = sestoji_processing.preprocess()
    sestoji_df.write_csv(PROCESSED_DIR / "sestoji_processed.csv")
    log.info(f"  [sestoji] {sestoji_df.shape[0]:,} rows × {sestoji_df.shape[1]} cols "
             f"→ sestoji_processed.csv")

    log.info("  [meritve] aggregating ARSO weather data to monthly level...")
    meritve_df = meritve_processing.preprocess()
    meritve_df.write_csv(PROCESSED_DIR / "vreme_mesecno.csv")
    log.info(f"  [meritve] {meritve_df.shape[0]:,} rows × {meritve_df.shape[1]} cols "
             f"→ vreme_mesecno.csv")

    return posek_df, odseki_df, sestoji_df, meritve_df


# ---------------------------------------------------------------------------
# Step 2 – Aggregate relationships
# ---------------------------------------------------------------------------

def step_aggregate():
    """
    Build relationship tables and return them as polars DataFrames.

    Sub-steps:
      a) najblizji_odseki_postaje – precompute nearest stations & neighbours
         (skipped when output file already exists)
      b) agg_posek_meritve  – posek × weather  (key: [ggo, odsek_id, leto_mesec])
      c) agg_posek_sosedi   – posek × neighbour posek  (key: [ggo, odsek_id, leto_mesec])
      d) agg_sestoji_odseke – odseki × sestoji  (key: [ggo, odsek_id])
    """
    log.info("=== Step 2: Aggregating relationships ===")

    najblizji_out = PROCESSED_DIR / "najblizji_odseki_postaje.csv"
    rebuild_najblizji = True
    if najblizji_out.exists():
        try:
            existing_cols = pl.read_csv(najblizji_out, n_rows=0).columns
            if "ggo" in existing_cols:
                rebuild_najblizji = False
                log.info("  [najblizji] output already exists with ggo key – skipping precomputation")
            else:
                log.info("  [najblizji] existing output missing ggo key – recomputing")
        except Exception:
            log.info("  [najblizji] existing output unreadable – recomputing")

    if rebuild_najblizji:
        log.info("  [najblizji] computing nearest stations & odsek neighbours "
                 "(this may take a while)...")
        najblizji_odseki_postaje_predracun.main()
        log.info(f"  [najblizji] written → {najblizji_out}")

    log.info("  [agg_posek_meritve] joining posek with rolling weather features...")
    agg_meritve_df = agg_posek_meritve.aggregate()
    log.info(f"  [agg_posek_meritve] {agg_meritve_df.shape[0]:,} rows × "
             f"{agg_meritve_df.shape[1]} cols")

    log.info("  [agg_posek_sosedi] joining posek with neighbour posek features...")
    agg_sosedi_df = agg_posek_sosedi.aggregate()
    log.info(f"  [agg_posek_sosedi] {agg_sosedi_df.shape[0]:,} rows × "
             f"{agg_sosedi_df.shape[1]} cols")

    log.info("  [agg_sestoji_odseke] joining odseki with aggregated sestoji...")
    agg_sestoji_df = agg_sestoji_odseke.aggregate()
    log.info(f"  [agg_sestoji_odseke] {agg_sestoji_df.shape[0]:,} rows × "
             f"{agg_sestoji_df.shape[1]} cols")

    return agg_meritve_df, agg_sosedi_df, agg_sestoji_df


# ---------------------------------------------------------------------------
# Step 3 – Join all tables
# ---------------------------------------------------------------------------

def step_join(
    posek_df: pl.DataFrame,
    agg_meritve_df: pl.DataFrame,
    agg_sosedi_df: pl.DataFrame,
    agg_sestoji_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Left-join all aggregated tables onto the posek feature table.

    Join order:
            posek  ×  agg_posek_meritve  on (ggo, odsek, leto_mesec)
             ×  agg_posek_sosedi   on (ggo, odsek, leto_mesec)
             ×  agg_odseki_sestoji on (ggo, odsek)        [static features]
    """
    log.info("=== Step 3: Joining datasets ===")

    base = posek_df
    if "ggo" not in base.columns:
        raise ValueError("posek_processed is missing 'ggo'; run updated posek_processing first.")

    # Determine ggo dtype from base and cast all agg tables to match
    ggo_dtype = base.schema["ggo"]

    def _align_ggo(df: pl.DataFrame) -> pl.DataFrame:
        if df.schema["ggo"] != ggo_dtype:
            df = df.with_columns(pl.col("ggo").cast(ggo_dtype))
        return df

    # agg tables use ['ggo', 'odsek_id'] as their composite key; align with posek's ['ggo', 'odsek']
    meritve_clean = _align_ggo(agg_meritve_df.rename({"odsek_id": "odsek"}))
    # drop 'used_station' — internal join-artifact, not a model feature
    if "used_station" in meritve_clean.columns:
        meritve_clean = meritve_clean.drop("used_station")
    base = base.join(meritve_clean, on=["ggo", "odsek", "leto_mesec"], how="left")
    log.info(f"  after weather join:   {base.shape[0]:,} rows × {base.shape[1]} cols")

    sosedi_clean = _align_ggo(agg_sosedi_df.rename({"odsek_id": "odsek"}))
    base = base.join(sosedi_clean, on=["ggo", "odsek", "leto_mesec"], how="left")
    log.info(f"  after neighbour join: {base.shape[0]:,} rows × {base.shape[1]} cols")

    sestoji_clean = _align_ggo(agg_sestoji_df.rename({"odsek_id": "odsek"}))
    base = base.join(sestoji_clean, on=["ggo", "odsek"], how="left", suffix="_odsek")
    log.info(f"  after odseki join:    {base.shape[0]:,} rows × {base.shape[1]} cols")

    # Fill nulls in numeric columns with 0; leave string/categorical as-is
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

    Boundaries (configure via module-level constants):
      train : leto  <  TRAIN_CUTOFF_YEAR
      val   : TRAIN_CUTOFF_YEAR  <= leto  <  VAL_CUTOFF_YEAR
      test  : leto  >= VAL_CUTOFF_YEAR

    Demo mode (--demo flag):
      Before splitting, the dataset is subsampled to DEMO_N_ODSEKI randomly
      chosen [ggo, odsek] pairs (seed=42) so the pipeline runs on weak hardware.
      Demo outputs go to data/demo/{train,val,test}.csv.

    Full outputs:
      data/train.csv, data/val.csv, data/test.csv
      data/processed/splits/{train,val,test}.csv  (traceability copies)
    """
    log.info("=== Step 4: Splitting and exporting ===")

    if "leto" not in df.columns:
        raise ValueError(
            "Column 'leto' not found in final dataset – cannot apply year-based split."
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
        demo_pairs = (
            df.select(["ggo", "odsek"]).unique().sample(n=n_sample, seed=42)
        )
        df = df.join(demo_pairs, on=["ggo", "odsek"], how="inner")

    train = df.filter(pl.col("leto") < TRAIN_CUTOFF_YEAR)
    val   = df.filter(
        (pl.col("leto") >= TRAIN_CUTOFF_YEAR) & (pl.col("leto") < VAL_CUTOFF_YEAR)
    )
    test  = df.filter(pl.col("leto") >= VAL_CUTOFF_YEAR)

    out_dir    = OUTPUT_DIR / "demo" if demo else OUTPUT_DIR
    splits_dir = PROCESSED_DIR / "splits" / ("demo" if demo else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split, name in [(train, "train"), (val, "val"), (test, "test")]:
        split.write_csv(out_dir    / f"{name}.csv")
        split.write_csv(splits_dir / f"{name}.csv")

    tag = " [demo]" if demo else ""
    log.info(f"  train{tag} : {len(train):,} rows → {out_dir}/train.csv")
    log.info(f"  val{tag}   : {len(val):,} rows → {out_dir}/val.csv")
    log.info(f"  test{tag}  : {len(test):,} rows → {out_dir}/test.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BarkWatch data pipeline – preprocess, aggregate, split."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            f"Create small demo datasets ({DEMO_N_ODSEKI} odseki) "
            "suitable for weak computers. Outputs go to data/demo/."
        ),
    )
    args = parser.parse_args()

    log.info("Starting BarkWatch data pipeline%s", " [DEMO MODE]" if args.demo else "")

    posek_df, odseki_df, sestoji_df, meritve_df = step_preprocess()
    agg_meritve_df, agg_sosedi_df, agg_sestoji_df = step_aggregate()
    final_df = step_join(posek_df, agg_meritve_df, agg_sosedi_df, agg_sestoji_df)
    step_split_export(final_df, demo=args.demo)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
