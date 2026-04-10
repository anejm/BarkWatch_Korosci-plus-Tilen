"""
Precompute nearest weather stations and nearest odseki for every odsek.

Outputs: data/processed/najblizji_odseki_postaje.csv
Columns:
    ggo                 – odsek class (part of composite key [ggo, odsek_id])
    odsek_id            – odsek identifier (part of composite key [ggo, odsek_id])
    station_123         – ID of nearest weather station (type 1, 2, or 3), all years
    station_23          – ID of nearest weather station (type 2 or 3), all years
    station_123_YYYY    – ID of nearest active station (type 1/2/3) for year YYYY
    station_23_YYYY     – ID of nearest active station (type 2/3) for year YYYY
                          (one pair of columns per year from 2006 to 2026;
                           NaN if no active station of that type exists for that year)
    bliznji_odseki      – semicolon-separated list of odsek IDs within RADIUS_KM

Usage:
    python najblizji_odseki_postaje_predracun.py
    python najblizji_odseki_postaje_predracun.py --radius 50
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
from shapely import wkt
from sklearn.neighbors import BallTree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent.parent.parent
ODSEKI_DIR         = _BASE / "data" / "processed"
LOKACIJE_PATH      = _BASE / "data" / "raw" / "ARSO" / "lokacije.csv"
POSTAJE_LETA_PATH  = ODSEKI_DIR / "postaje_po_letih.csv"
OUTPUT_PATH        = ODSEKI_DIR / "najblizji_odseki_postaje.csv"

YEARS = list(range(2006, 2027))

_ODSEKI_FILES = [
    ODSEKI_DIR / "odseki_processed_01.csv",
    ODSEKI_DIR / "odseki_processed_02.csv",
    ODSEKI_DIR / "odseki_processed_03.csv",
    ODSEKI_DIR / "odseki_processed_04.csv",
    ODSEKI_DIR / "odseki_processed_05.csv",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_RADIUS_KM = 1
EARTH_RADIUS_KM   = 6371.0

_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:3794", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_odseki_centroids() -> pd.DataFrame:
    """Load all odseki CSVs, compute WGS84 centroids and max border distance.

    Returns DataFrame with columns: ggo (str), odsek (str), lon, lat, max_border_km.
    max_border_km is the maximum distance from the centroid to any polygon
    vertex, in km (computed in EPSG:3794 metres)."""
    print("Loading odseki...")
    frames = []
    for path in _ODSEKI_FILES:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8", usecols=["ggo", "odsek", "geometry"])
            frames.append(df)
            print(f"  {path.name}: {len(df):,} rows")
        else:
            print(f"  {path.name}: NOT FOUND, skipping")

    if not frames:
        combined = ODSEKI_DIR / "odseki_processed.csv"
        if combined.exists():
            print(f"  No split files found; loading {combined.name} directly")
            df = pd.read_csv(combined, encoding="utf-8", usecols=["ggo", "odsek", "geometry"])
            frames.append(df)
            print(f"  {combined.name}: {len(df):,} rows")
        else:
            raise FileNotFoundError(
                "No odseki_processed split files or combined file found in "
                f"{ODSEKI_DIR}. Run step_preprocess() first."
            )

    df = pd.concat(frames, ignore_index=True)
    df["ggo"] = df["ggo"].astype(str)
    df["odsek"] = df["odsek"].astype(str)
    df = df.drop_duplicates(subset=["ggo", "odsek"]).reset_index(drop=True)
    print(f"  Total unique [ggo, odsek]: {len(df):,}")

    print("Computing centroids and max border distances...")
    geoms = df["geometry"].apply(wkt.loads)
    cx = geoms.apply(lambda g: g.centroid.x).values
    cy = geoms.apply(lambda g: g.centroid.y).values
    lons, lats = _TRANSFORMER.transform(cx, cy)

    def _max_border_km(geom, centroid_x, centroid_y) -> float:
        from shapely.geometry import MultiPolygon
        polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
        all_coords = np.concatenate([np.array(p.exterior.coords) for p in polys])
        dists = np.sqrt((all_coords[:, 0] - centroid_x) ** 2 + (all_coords[:, 1] - centroid_y) ** 2)
        return dists.max() / 1000.0

    max_border_km = np.array([
        _max_border_km(g, cx[i], cy[i]) for i, g in enumerate(geoms)
    ])

    df["lon"] = lons
    df["lat"] = lats
    df["max_border_km"] = max_border_km
    return df.drop(columns=["geometry"])


# ---------------------------------------------------------------------------
# Nearest station lookup
# ---------------------------------------------------------------------------

def _nearest_station_ids(
    odsek_df: pd.DataFrame,
    stations: pd.DataFrame,
    type_filter: list[int],
) -> np.ndarray:
    """Return array of station IDs (one per odsek) for the nearest station
    matching type_filter. Returns NaN for odseki when no matching station exists."""
    subset = stations[stations["Type"].isin(type_filter)].reset_index(drop=True)
    if subset.empty:
        return np.full(len(odsek_df), np.nan)

    station_coords = np.radians(subset[["Latitude", "Longitude"]].values)
    tree = BallTree(station_coords, metric="haversine")

    odsek_coords = np.radians(odsek_df[["lat", "lon"]].values)
    _, indices = tree.query(odsek_coords, k=1)

    return subset.iloc[indices.flatten()]["ID"].values


def _add_yearly_stations(
    odsek_df: pd.DataFrame,
    postaje_leta: pd.DataFrame,
) -> pd.DataFrame:
    """For each year in YEARS add two columns: station_123_YYYY and station_23_YYYY,
    containing the nearest active station of each type for that year.
    postaje_leta must have columns: leto, station_id, tip, Longitude, Latitude."""
    postaje_leta = postaje_leta.rename(columns={"station_id": "ID", "tip": "Type"})
    postaje_leta["Type"] = postaje_leta["Type"].astype(int)

    for year in YEARS:
        print(f"  Year {year}...", end=" ", flush=True)
        year_stations = postaje_leta[postaje_leta["leto"] == year]

        odsek_df[f"station_123_{year}"] = _nearest_station_ids(odsek_df, year_stations, [1, 2, 3])
        odsek_df[f"station_23_{year}"]  = _nearest_station_ids(odsek_df, year_stations, [2, 3])
        print("done")

    return odsek_df


# ---------------------------------------------------------------------------
# Nearest odseki within radius
# ---------------------------------------------------------------------------

def _bliznji_odseki_series(odsek_df: pd.DataFrame, radius_km: float) -> np.ndarray:
    """For each odsek return a semicolon-separated string of odsek IDs (excluding
    self) whose centroid falls within (radius_km + odsek_max_border_km + 1) of
    the source odsek's centroid.  The +1 km buffer ensures no true neighbour is
    missed even if the border estimate is slightly off."""
    print(f"  Building BallTree for {len(odsek_df):,} odseki...")
    coords = np.radians(odsek_df[["lat", "lon"]].values)
    tree = BallTree(coords, metric="haversine")

    # Per-odsek search radius: base + max extent of this odsek's polygon + 1 km buffer
    per_odsek_radii = (radius_km + odsek_df["max_border_km"].values + 1.0) / EARTH_RADIUS_KM
    print(f"  Querying neighbours (base radius {radius_km} km + per-odsek border + 1 km buffer)...")
    indices_list = tree.query_radius(coords, r=per_odsek_radii)

    odsek_ids = odsek_df["odsek"].values
    results = np.empty(len(odsek_df), dtype=object)
    for i, neighbours in enumerate(indices_list):
        neighbour_ids = [odsek_ids[j] for j in neighbours if j != i]
        results[i] = ";".join(neighbour_ids)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(radius_km: float = DEFAULT_RADIUS_KM) -> None:
    odsek_df = _load_odseki_centroids()

    print("Loading station locations...")
    lokacije = pd.read_csv(LOKACIJE_PATH, encoding="utf-8")
    print(f"  {len(lokacije)} stations loaded")

    print("Finding nearest station (type 1/2/3) across all years...")
    odsek_df["station_123"] = _nearest_station_ids(odsek_df, lokacije, [1, 2, 3])

    print("Finding nearest station (type 2/3) across all years...")
    odsek_df["station_23"] = _nearest_station_ids(odsek_df, lokacije, [2, 3])

    print("Loading active stations per year...")
    postaje_leta = pd.read_csv(POSTAJE_LETA_PATH, encoding="utf-8")
    print(f"  {len(postaje_leta):,} rows loaded ({postaje_leta['leto'].min()}–{postaje_leta['leto'].max()})")

    print("Finding nearest active station per year...")
    odsek_df = _add_yearly_stations(odsek_df, postaje_leta)

    print(f"Finding nearest odseki within {radius_km} km...")
    odsek_df["bliznji_odseki"] = _bliznji_odseki_series(odsek_df, radius_km)

    year_cols = [c for y in YEARS for c in (f"station_123_{y}", f"station_23_{y}")]
    out = odsek_df[["ggo", "odsek", "station_123", "station_23"] + year_cols + ["bliznji_odseki"]].rename(
        columns={"odsek": "odsek_id"}
    )

    print(f"Writing {len(out):,} rows to {OUTPUT_PATH} ...")
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute nearest stations and odseki for all odsek IDs."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_RADIUS_KM,
        help=f"Radius in km for neighbouring odseki (default: {DEFAULT_RADIUS_KM})",
    )
    args = parser.parse_args()
    main(radius_km=args.radius)
