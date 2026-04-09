"""
Find the 10 nearest odseki for a given odsek ID.

Usage:
    from bliznji_odseki import get_najblizje

    result = get_najblizje("31001")
    for o in result:
        print(o)   # #1 odsek=31002 (1.23 km)
"""

import math
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

import pandas as pd
from shapely import wkt
import pyproj

_BASE = Path(__file__).parent.parent.parent
ODSEKI_DIR = _BASE / "data" / "processed"

_ODSEKI_FILES = [
    ODSEKI_DIR / "odseki_processed_01.csv",
    ODSEKI_DIR / "odseki_processed_02.csv",
    ODSEKI_DIR / "odseki_processed_03.csv",
    ODSEKI_DIR / "odseki_processed_04.csv",
    ODSEKI_DIR / "odseki_processed_05.csv",
]

_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:3794", "EPSG:4326", always_xy=True)


@dataclass
class Odsek:
    rank: int
    id: str
    dist_km: float

    def __repr__(self):
        return f"#{self.rank} odsek={self.id} ({self.dist_km:.3f} km)"


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@lru_cache(maxsize=1)
def _load_odseki_centroids() -> pd.DataFrame:
    """Load all odseki CSVs, compute WGS84 centroids. Cached after first call."""
    frames = []
    for path in _ODSEKI_FILES:
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8", usecols=["odsek", "geometry"])
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    lons, lats = [], []
    for geom_str in df["geometry"]:
        geom = wkt.loads(geom_str)
        lon, lat = _TRANSFORMER.transform(geom.centroid.x, geom.centroid.y)
        lons.append(lon)
        lats.append(lat)

    df["lon"] = lons
    df["lat"] = lats
    return df.drop(columns=["geometry"])


def get_najblizje(odsek_id: str) -> list[Odsek]:
    """
    Return the 10 nearest odseki to the given odsek (excluding itself).

    Args:
        odsek_id: odsek code as string, e.g. "31001"

    Returns:
        List of 10 Odsek objects sorted by ascending distance.

    Raises:
        KeyError: if odsek_id not found
    """
    df = _load_odseki_centroids()

    rows = df[df["odsek"] == odsek_id]
    if rows.empty:
        raise KeyError(f"Odsek '{odsek_id}' not found.")

    lon0, lat0 = rows.iloc[0]["lon"], rows.iloc[0]["lat"]

    others = df[df["odsek"] != odsek_id].copy()
    others["dist_km"] = others.apply(
        lambda r: _haversine_km(lon0, lat0, r["lon"], r["lat"]),
        axis=1,
    )

    top10 = others.nsmallest(10, "dist_km")
    return [
        Odsek(rank=rank, id=row["odsek"], dist_km=round(float(row["dist_km"]), 3))
        for rank, (_, row) in enumerate(top10.iterrows(), start=1)
    ]


if __name__ == "__main__":
    import sys

    odsek = sys.argv[1] if len(sys.argv) > 1 else "31001"

    result = get_najblizje(odsek)
    print(f"\nOdsek: {odsek}")
    print("\nTop 10 najbližjih odsekov:")
    for o in result:
        print(f"  {o}")