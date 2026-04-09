"""
pipeline.py
-----------
Orchestrates the full BarkWatch data pipeline:

  1. Preprocess raw datasets  (posek, odsek, sestoj, meritve)
  2. Aggregate relationships  (target matrix, odseki+sestoji, weather per odsek,
                               monthly weather lookup, neighbour odseki features)
  3. Export train / val / test splits  (70 % / 15 % / 15 %, seed=42)

Usage:
    python src/pipeline.py

Outputs:
    data/train.csv
    data/val.csv
    data/test.csv
"""

import logging
import sys
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_DIR       = DATA_DIR

TARGET_CSV        = PROCESSED_DIR / "target.csv"
ODSEKI_AGG_CSV    = PROCESSED_DIR / "odseki_processed.csv"
POSEK_MERITVE_CSV = PROCESSED_DIR / "posek_meritve.csv"
VREME_MESECNO_CSV = PROCESSED_DIR / "vreme_mesecno.csv"

sys.path.insert(0, str(ROOT / "src" / "data_processing"))

import polars as pl


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def step1_preprocess() -> dict[str, pl.DataFrame | None]:
    """
    Call preprocess() on each raw-data module.

    Returns a dict with keys: posek, meritve, odsek, sestoj.
    Modules that cannot run (e.g. missing GeoPackage) return None.
    """
    log.info("═══ STEP 1: Preprocessing ═══════════════════════════════════")
    results: dict[str, pl.DataFrame | None] = {}

    log.info("  posek_processing.preprocess()")
    import posek_processing
    df = posek_processing.preprocess()
    log.info(f"    → {df.shape[0]:,} rows × {df.shape[1]} cols")
    results["posek"] = df

    log.info("  meritve_processing.preprocess()")
    import meritve_processing
    df = meritve_processing.preprocess()
    log.info(f"    → {df.shape[0]:,} rows × {df.shape[1]} cols")
    results["meritve"] = df

    log.info("  odseki_processing.preprocess()")
    try:
        import odseki_processing
        df = odseki_processing.preprocess()
        log.info(f"    → {df.shape[0]:,} rows × {df.shape[1]} cols")
        results["odsek"] = df
    except Exception as exc:
        log.warning(f"    skipped — {exc}")
        results["odsek"] = None

    log.info("  sestoji_processing.preprocess()")
    try:
        import sestoji_processing
        df = sestoji_processing.preprocess()
        log.info(f"    → {df.shape[0]:,} rows × {df.shape[1]} cols")
        results["sestoj"] = df
    except Exception as exc:
        log.warning(f"    skipped — {exc}")
        results["sestoj"] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _odsek_str(df: pl.DataFrame) -> pl.DataFrame:
    """Cast 'odsek' column to Utf8 for consistent join keys."""
    return df.with_columns(pl.col("odsek").cast(pl.Utf8))


def _ensure_agg_sestoji():
    """Run agg_sestoji_odseke.main() if its output CSV is missing."""
    if ODSEKI_AGG_CSV.exists():
        log.info("  odseki_processed.csv exists — skipping agg_sestoji_odseke")
        return
    log.info("  Running agg_sestoji_odseke.main() …")
    try:
        import agg_sestoji_odseke
        agg_sestoji_odseke.main()
        log.info(f"    → {ODSEKI_AGG_CSV}")
    except Exception as exc:
        log.warning(f"    agg_sestoji_odseke failed: {exc}")


def _ensure_posek_meritve():
    """Run agg_posek_meritve.main() if its output CSV is missing."""
    if POSEK_MERITVE_CSV.exists():
        log.info("  posek_meritve.csv exists — skipping agg_posek_meritve")
        return
    log.info("  Running agg_posek_meritve.main() (may take several minutes) …")
    try:
        import agg_posek_meritve
        agg_posek_meritve.main()
        log.info(f"    → {POSEK_MERITVE_CSV}")
    except Exception as exc:
        log.warning(f"    agg_posek_meritve failed: {exc}")


def _join_target(df: pl.DataFrame) -> pl.DataFrame:
    """Left-join 12-horizon targets on (odsek, leto_mesec)."""
    log.info("  Joining target (h1..h12) on (odsek, leto_mesec) …")
    import posek_processing
    df_target = posek_processing.make_target()
    df_target = _odsek_str(df_target)
    out = df.join(df_target, on=["odsek", "leto_mesec"], how="left")
    log.info(f"    shape: {out.shape}")
    return out


def _join_odseki(df: pl.DataFrame) -> pl.DataFrame:
    """Left-join aggregated odseki + sestoji static features on odsek."""
    if not ODSEKI_AGG_CSV.exists():
        log.warning("  odseki_processed.csv missing — skipping odseki join")
        return df
    log.info("  Joining odseki forest features on odsek …")
    df_odseki = pl.read_csv(ODSEKI_AGG_CSV, infer_schema_length=10_000)
    if "geometry" in df_odseki.columns:
        df_odseki = df_odseki.drop("geometry")
    df_odseki = _odsek_str(df_odseki)
    out = df.join(df_odseki, on="odsek", how="left")
    log.info(f"    shape: {out.shape}")
    return out


def _join_posek_meritve(df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join posek weather window features (30d / 90d / 365d) aggregated
    to (odsek, leto_mesec) level.
    """
    if not POSEK_MERITVE_CSV.exists():
        log.warning("  posek_meritve.csv missing — skipping weather window join")
        return df
    log.info("  Joining posek_meritve weather features on (odsek, leto_mesec) …")
    df_pm = pl.read_csv(POSEK_MERITVE_CSV, infer_schema_length=10_000)
    df_pm = _odsek_str(df_pm)

    # Extract YYYY-MM from the posekano date string
    if "posekano" in df_pm.columns:
        df_pm = df_pm.with_columns(
            pl.col("posekano").cast(pl.Utf8).str.slice(0, 7).alias("leto_mesec")
        )

    # Only average the weather window columns
    feat_cols = [c for c in df_pm.columns
                 if c.startswith(("30d_", "90d_", "365d_"))]
    if not feat_cols:
        log.warning("    no window feature columns found in posek_meritve.csv")
        return df

    df_pm_monthly = df_pm.group_by(["odsek", "leto_mesec"]).agg(
        [pl.col(c).mean() for c in feat_cols]
    )
    out = df.join(df_pm_monthly, on=["odsek", "leto_mesec"], how="left")
    log.info(f"    shape: {out.shape}")
    return out


def _join_vreme_monthly(df: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join monthly weather from the nearest ARSO station per odsek
    on (odsek, leto_mesec).

    Resolves nearest station via bliznje_vremenske_postaje, then attaches
    monthly aggregated weather from vreme_mesecno.csv.
    """
    if not VREME_MESECNO_CSV.exists():
        log.warning("  vreme_mesecno.csv missing — skipping monthly weather join")
        return df

    log.info("  Resolving nearest ARSO station per odsek …")
    try:
        from bliznje_vremenske_postaje import get_postaje, _load_vreme
    except Exception as exc:
        log.warning(f"  bliznje_vremenske_postaje unavailable: {exc}")
        return df

    try:
        vreme_raw = _load_vreme()
        available_sids = set(vreme_raw.index.get_level_values("station_id"))
    except Exception as exc:
        log.warning(f"  Could not load vreme index: {exc}")
        return df

    unique_odseki = df["odsek"].unique().to_list()
    odsek_station: dict[str, str | None] = {}
    for oid in unique_odseki:
        try:
            postaje = get_postaje(str(oid))
            sid = None
            for p in postaje["all"]:
                if str(p.id) in available_sids:
                    sid = str(p.id)
                    break
            odsek_station[oid] = sid
        except Exception:
            odsek_station[oid] = None

    n_mapped = sum(1 for v in odsek_station.values() if v is not None)
    log.info(f"    {n_mapped}/{len(unique_odseki)} odseki mapped to ARSO stations")

    if n_mapped == 0:
        log.warning("  No odseki mapped — skipping monthly weather join")
        return df

    mapping_rows = [{"odsek": str(k), "station_id": v}
                    for k, v in odsek_station.items() if v is not None]
    df_map = pl.DataFrame(mapping_rows)

    df_vreme = pl.read_csv(VREME_MESECNO_CSV, infer_schema_length=10_000)
    df_vreme = df_vreme.with_columns(pl.col("station_id").cast(pl.Utf8))

    # Prefix all non-key columns to avoid clashes with posek feature columns
    rename_map = {c: f"vreme_{c}"
                  for c in df_vreme.columns
                  if c not in ("station_id", "leto_mesec")}
    df_vreme = df_vreme.rename(rename_map)

    df_vreme_odsek = (
        df_map
        .join(df_vreme, on="station_id", how="left")
        .drop("station_id")
    )
    out = df.join(df_vreme_odsek, on=["odsek", "leto_mesec"], how="left")
    log.info(f"    shape: {out.shape}")
    return out


def _join_bliznji_odseki(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add aggregated forest characteristics from the 10 nearest neighbour
    odseki per odsek, using bliznji_odseki.get_najblizje() and
    agg_posek_sosedi.agg_odseki().
    """
    log.info("  Computing bliznji odseki neighbour features …")
    try:
        from bliznji_odseki import get_najblizje
        from agg_posek_sosedi import agg_odseki
    except Exception as exc:
        log.warning(f"  bliznji_odseki / agg_posek_sosedi unavailable: {exc}")
        return df

    unique_odseki = df["odsek"].unique().to_list()
    rows: list[dict] = []
    skipped = 0

    for oid in unique_odseki:
        try:
            neighbours = get_najblizje(str(oid))
            agg = agg_odseki(neighbours)
            row: dict = {"odsek": str(oid)}
            for k, v in agg.items():
                row[f"sosedi_{k}"] = float(v) if v is not None else None
            rows.append(row)
        except Exception:
            skipped += 1

    if not rows:
        log.warning("  No neighbour features computed — skipping")
        return df

    log.info(f"    {len(rows)} odseki aggregated, {skipped} skipped")
    df_sosedi = pl.from_pandas(pd.DataFrame(rows))
    out = df.join(df_sosedi, on="odsek", how="left")
    log.info(f"    shape: {out.shape}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Aggregation orchestration
# ─────────────────────────────────────────────────────────────────────────────

def step2_aggregate(df_posek: pl.DataFrame) -> pl.DataFrame:
    """
    Join all relationship DataFrames onto the posek feature table.

    Join order (all left joins):
      posek → target          on (odsek, leto_mesec)
      posek → odseki          on  odsek
      posek → posek_meritve   on (odsek, leto_mesec)
      posek → vreme_mesecno   on (odsek, leto_mesec)   via nearest station
      posek → bliznji_odseki  on  odsek                via neighbour aggregation
    """
    log.info("═══ STEP 2: Aggregation ══════════════════════════════════════")

    _ensure_agg_sestoji()
    _ensure_posek_meritve()

    df = _odsek_str(df_posek)
    df = _join_target(df)
    df = _join_odseki(df)
    df = _join_posek_meritve(df)
    df = _join_vreme_monthly(df)
    df = _join_bliznji_odseki(df)

    log.info(f"  Final combined shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Split and export
# ─────────────────────────────────────────────────────────────────────────────

def step3_export(df: pl.DataFrame) -> None:
    """
    Shuffle deterministically and split into train / val / test (70/15/15).
    Saves three CSV files to data/.
    """
    log.info("═══ STEP 3: Split and export ═════════════════════════════════")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(df)

    df = df.sample(fraction=1.0, seed=42, shuffle=True)

    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    df_train = df.slice(0, n_train)
    df_val   = df.slice(n_train, n_val)
    df_test  = df.slice(n_train + n_val)

    df_train.write_csv(OUT_DIR / "train.csv")
    df_val.write_csv(OUT_DIR / "val.csv")
    df_test.write_csv(OUT_DIR / "test.csv")

    log.info(f"  train : {len(df_train):,} rows  →  {OUT_DIR / 'train.csv'}")
    log.info(f"  val   : {len(df_val):,} rows  →  {OUT_DIR / 'val.csv'}")
    log.info(f"  test  : {len(df_test):,} rows  →  {OUT_DIR / 'test.csv'}")


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("BarkWatch data pipeline starting …")
    preprocessed  = step1_preprocess()
    df_combined   = step2_aggregate(preprocessed["posek"])
    step3_export(df_combined)
    log.info("Pipeline complete.")
