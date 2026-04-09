import geopandas as gpd
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = str(_ROOT / "data" / "raw" / "ZGS" / "odseki_gozdno.gpkg")
OUTPUT_FILE = str(_ROOT / "data" / "processed" / "odseki_processed.csv")

COLUMNS_TO_KEEP = [
    'ggo',
    'odsek',
    'povrsina',
    'katgozd',
    'ohranjen',
    'relief',
    'nagib',
    'nvod',
    'nvdo',
    'tarifa_sm',
    'tarifa_je',
    'tarifa_oi',
    'tarifa_bu',
    'tarifa_hr',
    'tarifa_pl',
    'tarifa_tl',
    'tarifa_ml',
    'spravilo',
    'razdalja',
    'odprtost',
    'pozar',
    'intgosp',
    'grt1',
    'rk_gurs',
    'carb_tot_c',
    'ponor_c',
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
    Read odseki GeoPackage and return a cleaned polars DataFrame.

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
