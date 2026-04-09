import geopandas as gpd
import pandas as pd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE  = str(_ROOT / "data" / "raw" / "ZGS" / "odseki_gozdno.gpkg")
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
    'carb_tot_t',
    'ponor_c',
]

ONEHOT_COLUMNS = [
    'ggo',
    'katgozd',
    'ohranjen',
    'relief',
    'spravilo',
    'pozar',
    'intgosp',
    'rk_gurs',
]


def preprocess() -> pd.DataFrame:
    """
    Load and process odseki GeoPackage, return as pandas DataFrame.

    Applies column selection and one-hot encoding.

    Raises:
        FileNotFoundError: if the GeoPackage source file does not exist.
    """
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"GeoPackage not found: {INPUT_FILE}")

    gdf = gpd.read_file(INPUT_FILE)
    available = [col for col in COLUMNS_TO_KEEP if col in gdf.columns]
    missing = [col for col in COLUMNS_TO_KEEP if col not in gdf.columns]
    if missing:
        print(f"WARNING: columns not found and skipped: {missing}")

    df = gdf[available].copy()

    onehot_existing = [col for col in ONEHOT_COLUMNS if col in df.columns]
    df = pd.get_dummies(df, columns=onehot_existing, dummy_na=False)

    return df


def main():
    print(f"Reading {INPUT_FILE}...")
    gdf = gpd.read_file(INPUT_FILE)

    print(f"Total rows: {len(gdf)}")
    print(f"Available columns: {list(gdf.columns)}")

    available = [col for col in COLUMNS_TO_KEEP if col in gdf.columns]
    missing = [col for col in COLUMNS_TO_KEEP if col not in gdf.columns]

    if missing:
        print(f"\nWARNING: These columns were not found in the file and will be skipped: {missing}")

    print(f"\nKeeping {len(available)} columns: {available}")

    df = gdf[available].copy()

    onehot_existing = [col for col in ONEHOT_COLUMNS if col in df.columns]
    print(f"\nApplying one-hot encoding to: {onehot_existing}")
    df = pd.get_dummies(df, columns=onehot_existing, dummy_na=False)

    print(f"\nNull counts per column:")
    print(df.isnull().sum())

    print(f"\nBasic stats:")
    print(df.describe())

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nSaved to {OUTPUT_FILE} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == '__main__':
    main()
