"""
pipeline.py
-----------
Orchestrates the full BarkWatch data pipeline:

  1. Preprocess  – clean raw sources (posek, odseki, sestoji, meritve)
  2. Aggregate   – build relationship tables (weather join, neighbour join,
                   odseki+sestoji join)
  3. Join        – combine all tables into one feature-rich dataset
  4. Export      – split into train / val / test and write CSVs

Enrichment flags (all off by default):
  --weather   add rolling 12-month weather features per parcel
  --sosedi    add neighbour posek features per parcel
  --odseki    add static forest-segment (sestoji) features per parcel

Usage:
    python pipeline.py                                # base features only
    python pipeline.py --weather --sosedi --odseki   # fully enriched
    python pipeline.py --demo                         # small demo dataset (N odseki)
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
import postaje_po_letih
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
      data/processed/posek_processed.csv   — feature table (no raw target col)
      data/processed/target.csv            — 12-horizon target matrix (log1p)
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

    log.info("  [posek] building 12-horizon target matrix...")
    target_df = posek_processing.make_target_kubikov()
    target_df.write_csv(PROCESSED_DIR / "target.csv")
    log.info(f"  [posek] {target_df.shape[0]:,} rows × {target_df.shape[1]} cols "
             f"→ target.csv")

    log.info("  [odseki] loading forest compartment GeoPackage...")
    odseki_df = odseki_processing.preprocess()
    odseki_df.write_csv(PROCESSED_DIR / "odseki_processed.csv")
    log.info(f"  [odseki] {odseki_df.shape[0]:,} rows × {odseki_df.shape[1]} cols "
             f"→ odseki_processed.csv")
    odseki_processing.save_geometry()
    log.info("  [odseki] geometry saved → odseki_geometry.csv")

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

    log.info("  [postaje] computing active stations per year...")
    postaje_po_letih.main()
    log.info("  [postaje] → postaje_po_letih.csv")

    return posek_df, odseki_df, sestoji_df, meritve_df


# ---------------------------------------------------------------------------
# Step 2 – Aggregate relationships
# ---------------------------------------------------------------------------

def step_aggregate(
    weather: bool,
    sosedi: bool,
    odseki: bool,
) -> dict[str, Optional[pl.DataFrame]]:
    """
    Build the requested relationship tables and return them in a dict.

    Keys always present (value is None when the flag was not set):
      "meritve"  – rolling 12-month weather features
      "sosedi"   – neighbour posek features
      "sestoji"  – static odsek/segment features

    When --weather or --sosedi is requested, najblizji_odseki_postaje is
    precomputed first (skipped when a valid output already exists).
    """
    log.info("=== Step 2: Aggregating relationships ===")

    result: dict[str, Optional[pl.DataFrame]] = {
        "meritve": None,
        "sosedi":  None,
        "sestoji": None,
    }

    if weather or sosedi:
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

    if weather:
        log.info("  [agg_posek_meritve] joining posek with rolling weather features...")
        result["meritve"] = agg_posek_meritve.aggregate()
        log.info(f"  [agg_posek_meritve] {result['meritve'].shape[0]:,} rows × "
                 f"{result['meritve'].shape[1]} cols")
    else:
        log.info("  [agg_posek_meritve] skipped (--weather not set)")

    if sosedi:
        log.info("  [agg_posek_sosedi] joining posek with neighbour posek features...")
        result["sosedi"] = agg_posek_sosedi.aggregate()
        log.info(f"  [agg_posek_sosedi] {result['sosedi'].shape[0]:,} rows × "
                 f"{result['sosedi'].shape[1]} cols")
    else:
        log.info("  [agg_posek_sosedi] skipped (--sosedi not set)")

    if odseki:
        log.info("  [agg_sestoji_odseke] joining odseki with aggregated sestoji...")
        result["sestoji"] = agg_sestoji_odseke.aggregate()
        log.info(f"  [agg_sestoji_odseke] {result['sestoji'].shape[0]:,} rows × "
                 f"{result['sestoji'].shape[1]} cols")
    else:
        log.info("  [agg_sestoji_odseke] skipped (--odseki not set)")

    return result


# ---------------------------------------------------------------------------
# Step 3 – Join all tables
# ---------------------------------------------------------------------------

def step_join(
    posek_df: pl.DataFrame,
    agg: dict[str, Optional[pl.DataFrame]],
) -> pl.DataFrame:
    """
    Left-join the requested aggregated tables onto the posek feature table.
    Any agg entry that is None is silently skipped.

    Join order (when present):
      posek  ×  weather          on (ggo, odsek, leto_mesec)
             ×  neighbours       on (ggo, odsek, leto_mesec)
             ×  odseki/sestoji   on (ggo, odsek)   [static features]
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
    if agg["meritve"] is not None:
        meritve_clean = _align_ggo(agg["meritve"].rename({"odsek_id": "odsek"}))
        if "used_station" in meritve_clean.columns:
            meritve_clean = meritve_clean.drop("used_station")
        base = base.join(meritve_clean, on=["ggo", "odsek", "leto_mesec"], how="left")
        log.info(f"  after weather join:   {base.shape[0]:,} rows × {base.shape[1]} cols")

    if agg["sosedi"] is not None:
        sosedi_clean = _align_ggo(agg["sosedi"].rename({"odsek_id": "odsek"}))
        base = base.join(sosedi_clean, on=["ggo", "odsek", "leto_mesec"], how="left")
        log.info(f"  after neighbour join: {base.shape[0]:,} rows × {base.shape[1]} cols")

    if agg["sestoji"] is not None:
        sestoji_clean = _align_ggo(agg["sestoji"].rename({"odsek_id": "odsek"}))
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
        if "leto_mesec" not in df.columns:
            raise ValueError(
                "Neither 'leto' nor 'leto_mesec' found in final dataset – cannot apply year-based split."
            )
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
        description="BarkWatch data pipeline – preprocess, aggregate, split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python pipeline.py                                # base features only\n"
            "  python pipeline.py --weather --sosedi --odseki   # fully enriched\n"
            "  python pipeline.py --sosedi                       # neighbours only\n"
        ),
    )
    parser.add_argument("--demo", action="store_true",
                        help=f"subsample to {DEMO_N_ODSEKI} odseki (weak hardware)")
    parser.add_argument("--weather", action="store_true",
                        help="add rolling 12-month weather features per parcel")
    parser.add_argument("--sosedi", action="store_true",
                        help="add neighbour posek features per parcel")
    parser.add_argument("--odseki", action="store_true",
                        help="add static forest-segment (sestoji) features per parcel")
    args = parser.parse_args()

    enrichments = [
        "weather" if args.weather else "",
        "sosedi"  if args.sosedi  else "",
        "odseki"  if args.odseki  else "",
    ]
    active = [e for e in enrichments if e]
    label = f" [{', '.join(active)}]" if active else " [base only]"
    log.info("Starting BarkWatch data pipeline%s%s",
             " [DEMO MODE]" if args.demo else "", label)

    posek_df, odseki_df, sestoji_df, meritve_df = step_preprocess()
    agg = step_aggregate(weather=args.weather, sosedi=args.sosedi, odseki=args.odseki)
    final_df = step_join(posek_df, agg)
    step_split_export(final_df, demo=args.demo)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
