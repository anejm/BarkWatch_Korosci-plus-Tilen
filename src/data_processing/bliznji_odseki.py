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
    if not frames:
        combined = ODSEKI_DIR / "odseki_processed.csv"
        if combined.exists():
            df = pd.read_csv(combined, encoding="utf-8", usecols=["odsek", "geometry"])
            frames.append(df)
        else:
            raise FileNotFoundError(
                "No odseki_processed split files or combined file found in "
                f"{ODSEKI_DIR}. Run step_preprocess() first."
            )
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


def get_najblizje(odsek_id: str, max_km: float | None = None) -> list[str]:
    """
    Return a list of odsek IDs: the given odsek itself plus nearby ones.

    Args:
        odsek_id: odsek code as string, e.g. "31001"
        max_km:   optional radius in km. If given, returns all odseki within
                  that distance. If None (default), returns the 10 nearest.

    Returns:
        List of odsek ID strings — first entry is always odsek_id itself,
        followed by neighbours sorted by ascending distance.

    Raises:
        KeyError:   if odsek_id not found
        ValueError: if max_km is negative
    """
    if max_km is not None and max_km < 0:
        raise ValueError(f"max_km mora biti nenegativen, dobili smo: {max_km}")

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

    if max_km is None:
        neighbours = others.nsmallest(10, "dist_km")
    else:
        neighbours = others[others["dist_km"] <= max_km].sort_values("dist_km")

    return [odsek_id] + neighbours["odsek"].tolist()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python bliznji_odseki.py <odsek_id> [max_km]")
        sys.exit(1)

    odsek = sys.argv[1]
    max_km = float(sys.argv[2]) if len(sys.argv) > 2 else None

    result = get_najblizje(odsek, max_km=max_km)
    label = f"v radiju {max_km} km" if max_km is not None else "10 najbližjih"
    print(f"\nOdsek + {label} ({len(result) - 1} sosedov): {result}")