"""
Combines odseki_gozdno.gpkg and sestoji.gpkg into odseki.csv.
Sestoji are aggregated by 'odsek' (area-weighted averages) and joined to odseki.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("data/raw/ZGS")
OUT_PATH = Path("data/processed/odseki.csv")

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

# Sestoji columns to aggregate (exclude odsek, sestoj, geometry, povrsina)
SESTOJI_AGG_COLS = [
    'lzskdv11', 'lzskdv11_m', 'lzskdv21', 'lzskdv21_m',
    'lzskdv30', 'lzskdv30_m', 'lzskdv34', 'lzskdv34_m',
    'lzskdv39', 'lzskdv39_m', 'lzskdv41', 'lzskdv41_m',
    'lzskdv50', 'lzskdv50_m', 'lzskdv60', 'lzskdv60_m',
    'lzskdv70', 'lzskdv70_m', 'lzskdv80', 'lzskdv80_m',
    'negovan', 'pompov', 'lzigl', 'lzlst', 'lzsku',
    'etigl', 'etlst', 'etsku',
]


def load_gpkg(path: Path, cols: list) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    available = [c for c in cols if c in gdf.columns]
    missing = set(cols) - set(available)
    if missing:
        print(f"  Warning: columns not found in {path.name}: {missing}")
    return gdf[available]


def weighted_mean(group: pd.DataFrame, value_cols: list, weight_col: str) -> pd.Series:
    weights = group[weight_col].fillna(0)
    total_weight = weights.sum()
    result = {}
    for col in value_cols:
        if col not in group.columns:
            continue
        vals = pd.to_numeric(group[col], errors='coerce')
        if total_weight > 0:
            result[col] = (vals * weights).sum() / total_weight
        else:
            result[col] = np.nan
    return pd.Series(result)


def aggregate_sestoji(sestoji: gpd.GeoDataFrame) -> pd.DataFrame:
    agg_cols = [c for c in SESTOJI_AGG_COLS if c in sestoji.columns]

    def agg_group(group):
        return weighted_mean(group, agg_cols, 'povrsina')

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

    print("Aggregating sestoji by odsek (area-weighted mean)...")
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
