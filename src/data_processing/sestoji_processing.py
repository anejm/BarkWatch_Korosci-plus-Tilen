import geopandas as gpd
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = str(_ROOT / "data" / "raw" / "ZGS" / "sestoji.gpkg")
OUTPUT_FILE = str(_ROOT / "data" / "processed" / "sestoji_processed.csv")

COLUMNS_TO_KEEP = [
    'ggo',
    'odsek',
    'sestoj',
    'povrsina',
    'rfaza',
    'lzskdv11',
    'lzskdv11_m',
    'lzskdv21',
    'lzskdv21_m',
    'lzskdv30',
    'lzskdv34',
    'lzskdv39',
    'lzskdv41',
    'lzskdv50',
    'lzskdv60',
    'lzskdv70',
    'lzskdv80',
    'sklep',
    'zasnova',
    'negovan',
    'pompov',
    'pomzas',
    'lzigl',
    'lzlst',
    'lzsku',
    'etigl',
    'etlst',
    'etsku',
]

def main():
    print(f"Reading {INPUT_FILE}...")
    gdf = gpd.read_file(INPUT_FILE)

    print(f"Total rows: {len(gdf)}")
    print(f"Available columns: {list(gdf.columns)}")

    # Check which of our desired columns actually exist in the file
    available = [col for col in COLUMNS_TO_KEEP if col in gdf.columns]
    missing = [col for col in COLUMNS_TO_KEEP if col not in gdf.columns]

    if missing:
        print(f"\nWARNING: These columns were not found in the file and will be skipped: {missing}")

    print(f"\nKeeping {len(available)} columns: {available}")

    df = gdf[available].copy()

    print(f"\nNull counts per column:")
    print(df.isnull().sum())

    print(f"\nBasic stats:")
    print(df.describe())

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nSaved to {OUTPUT_FILE} ({len(df)} rows, {len(df.columns)} columns)")

def preprocess() -> "pl.DataFrame":
    """
    Read sestoji GeoPackage and return a cleaned polars DataFrame.

    Raises:
        FileNotFoundError: if the GeoPackage source file does not exist.
    """
    import polars as pl
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"GeoPackage not found: {INPUT_FILE}")
    gdf = gpd.read_file(INPUT_FILE)
    available = [col for col in COLUMNS_TO_KEEP if col in gdf.columns]
    df = gdf[available].copy()
    return pl.from_pandas(df)


if __name__ == '__main__':
    main()
