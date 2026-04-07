"""
Combines odseki_gozdno.gpkg and sestoji.gpkg into odseki.csv.
Sestoji are aggregated by 'odsek' and joined to odseki.

Aggregation strategy per column type:
  - Percentage columns (lzskdv*): area-weighted average (uteženo s povrsino)
      Ratios must be weighted by the area of each sestoj to yield a correct
      composite percentage for the whole odsek.
  - Absolute volume/area columns: sum across sestoji in the odsek
      These are already in absolute units (m3 or ha), so summing gives the
      correct total for the odsek.

Column reference (source: ZGS sestoji layer):
  odsek         – šifra odseka
  sestoj        – šifra sestoja
  povrsina      – površina sestoja [ha]

  --- Deleži drevesnih vrst (area-weighted average [%]) ---
  lzskdv11      – delež smreke [%]
  lzskdv11_m   – delež smreke v mladju [%]
  lzskdv21      – delež jelke [%]
  lzskdv21_m   – delež jelke v mladju [%]
  lzskdv30      – delež bora [%]
  lzskdv30_m   – delež bora v mladju [%]
  lzskdv34      – delež macesna [%]
  lzskdv34_m   – delež macesna v mladju [%]
  lzskdv39      – delež ostalih iglavcev [%]
  lzskdv39_m   – delež ostalih iglavcev v mladju [%]
  lzskdv41      – delež bukve [%]
  lzskdv41_m   – delež bukve v mladju [%]
  lzskdv50      – delež hrastov [%]
  lzskdv50_m   – delež hrastov v mladju [%]
  lzskdv60      – delež plemenitih listavcev [%]
  lzskdv60_m   – delež plemenitih listavcev v mladju [%]
  lzskdv70      – delež trdih listavcev [%]
  lzskdv70_m   – delež trdih listavcev v mladju [%]
  lzskdv80      – delež mehkih listavcev [%]
  lzskdv80_m   – delež mehkih listavcev v mladju [%]

  --- Negovanost (area-weighted average, ordinalna šifra) ---
  negovan       – šifra negovanosti sestoja (ordinalna vrednost)

  --- Absolutne vrednosti (sum across sestoji) ---
  pompov        – površina pomladka v sestoju [ha]
  lzigl         – lesna zaloga iglavcev [m3]
  lzlst         – lesna zaloga listavcev [m3]
  lzsku         – skupna lesna zaloga [m3]
  etigl         – možni posek iglavcev [m3]
  etlst         – možni posek listavcev [m3]
  etsku         – skupni možni posek [m3]
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/ZGS")
OUT_PATH = Path("data/processed/odseki_processed.csv")

ODSEKI_COLS = [
    'ggo', 'odsek', 'povrsina', 'rgr_ggo', 'katgozd', 'ohranjen',
    'polpokr', 'relief', 'lega', 'nagib', 'nvod', 'nvdo', 'kamnina',
    'kamnit', 'skalnat',
    'tarifa_sm', 'tarifa_je', 'tarifa_oi', 'tarifa_bu', 'tarifa_hr',
    'tarifa_pl', 'tarifa_tl', 'tarifa_ml',
    'odprtost', 'pozar', 'intgosp', 'grt1', 'rk', 'geometry',
]

SESTOJI_COLS = [
    'odsek', 'sestoj', 'povrsina', 'lzskdv11', 'lzskdv11_m', 'lzskdv21',
    'lzskdv21_m', 'lzskdv30', 'lzskdv30_m', 'lzskdv34', 'lzskdv34_m',
    'lzskdv39', 'lzskdv39_m', 'lzskdv41', 'lzskdv41_m', 'lzskdv50',
    'lzskdv50_m', 'lzskdv60', 'lzskdv60_m', 'lzskdv70', 'lzskdv70_m',
    'lzskdv80', 'lzskdv80_m', 'negovan', 'pompov', 'lzigl', 'lzlst',
    'lzsku', 'etigl', 'etlst', 'etsku', 'geometry',
]

# Percentage/ratio columns: aggregate with area-weighted average
WEIGHTED_MEAN_COLS = [
    'lzskdv11', 'lzskdv11_m', 'lzskdv21', 'lzskdv21_m',
    'lzskdv30', 'lzskdv30_m', 'lzskdv34', 'lzskdv34_m',
    'lzskdv39', 'lzskdv39_m', 'lzskdv41', 'lzskdv41_m',
    'lzskdv50', 'lzskdv50_m', 'lzskdv60', 'lzskdv60_m',
    'lzskdv70', 'lzskdv70_m', 'lzskdv80', 'lzskdv80_m',
    'negovan',
]

# Absolute value columns: aggregate with sum (m3 or ha)
SUM_COLS = [
    'pompov', 'lzigl', 'lzlst', 'lzsku',
    'etigl', 'etlst', 'etsku',
]


def load_gpkg(path: Path, cols: list) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    available = [c for c in cols if c in gdf.columns]
    missing = set(cols) - set(available)
    if missing:
        print(f"  Warning: columns not found in {path.name}: {missing}")
    return gdf[available]


def aggregate_sestoji(sestoji: gpd.GeoDataFrame) -> pd.DataFrame:
    wm_cols = [c for c in WEIGHTED_MEAN_COLS if c in sestoji.columns]
    sum_cols = [c for c in SUM_COLS if c in sestoji.columns]

    def agg_group(group):
        weights = group['povrsina'].fillna(0)
        total_weight = weights.sum()
        result = {}

        # Area-weighted average for percentage/ratio columns
        for col in wm_cols:
            vals = pd.to_numeric(group[col], errors='coerce')
            if total_weight > 0:
                result[col] = (vals * weights).sum() / total_weight
            else:
                result[col] = np.nan

        # Sum for absolute value columns (m3, ha)
        for col in sum_cols:
            vals = pd.to_numeric(group[col], errors='coerce')
            result[col] = vals.sum(min_count=1)

        return pd.Series(result)

    aggregated = sestoji.groupby('odsek').apply(agg_group).reset_index()
    return aggregated


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading odseki_gozdno.gpkg...")
    odseki = load_gpkg(RAW_DIR / "odseki_gozdno.gpkg", ODSEKI_COLS)
    print(f"  {len(odseki)} rows, columns: {list(odseki.columns)}")

    print("Loading sestoji.gpkg...")
    sestoji = load_gpkg(RAW_DIR / "sestoji.gpkg", SESTOJI_COLS)
    print(f"  {len(sestoji)} rows, columns: {list(sestoji.columns)}")

    print("Aggregating sestoji by odsek...")
    sestoji_agg = aggregate_sestoji(sestoji)
    print(f"  {len(sestoji_agg)} unique odsek values after aggregation")

    print("Joining odseki with aggregated sestoji on 'odsek'...")
    odseki_df = pd.DataFrame(odseki.drop(columns='geometry', errors='ignore'))
    # Preserve WKT geometry
    if 'geometry' in odseki.columns:
        odseki_df['geometry'] = odseki['geometry'].to_wkt()

    merged = odseki_df.merge(sestoji_agg, on='odsek', how='left', suffixes=('_odsek', '_sestoj'))

    # Rename ambiguous povrsina columns if both exist
    if 'povrsina_odsek' in merged.columns:
        merged.rename(columns={'povrsina_odsek': 'povrsina'}, inplace=True)

    print(f"  {len(merged)} rows after join, {merged.columns.tolist()}")

    print(f"Saving to {OUT_PATH}...")
    merged.to_csv(OUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
