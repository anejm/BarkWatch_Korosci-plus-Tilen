"""
Lookup top 3 weather stations and weather data for a given odsek ID.
Computes nearest stations on-the-fly from odseki geometry and station coordinates.

Usage:
    from bliznje_vremenske_postaje import get_postaje, get_vreme

    # nearest stations
    result = get_postaje("31001")
    print(result["temp"])   # top 3 temperature stations (type 2+3)
    print(result["all"])    # top 3 all-data stations (type 1+2+3)

    # weather data for date range
    vreme = get_vreme("31001", "2024-01-01", "2024-01-31")
    print(vreme["temp"])    # DataFrame from nearest temp station
    print(vreme["all"])     # DataFrame from nearest all-data station
"""

import math
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from datetime import date

import pandas as pd
from shapely import wkt
import pyproj

_BASE = Path(__file__).parent.parent.parent
ODSEKI_DIR   = _BASE / "data" / "processed"
LOKACIJE_PATH = _BASE / "data" / "raw" / "ARSO" / "lokacije.csv"
VREME_PATH    = _BASE / "data" / "raw" / "ARSO" / "vreme.csv"

_ODSEKI_FILES = [
    ODSEKI_DIR / "odseki_processed_01.csv",
    ODSEKI_DIR / "odseki_processed_02.csv",
    ODSEKI_DIR / "odseki_processed_03.csv",
    ODSEKI_DIR / "odseki_processed_04.csv",
    ODSEKI_DIR / "odseki_processed_05.csv",
]

# Odseki geometry CRS → WGS84
_TRANSFORMER = pyproj.Transformer.from_crs("EPSG:3794", "EPSG:4326", always_xy=True)

# Mapping of garbled column names → clean names (double-encoded UTF-8 artifact)
_COL_RENAME = {
    "povp_dnevna_T_ÂdegC":      "povp_T_degC",
    "max_T_ÂdegC":              "max_T_degC",
    "min_T_ÂdegC":              "min_T_degC",
    "koliÄ\x8dina_padavin_mm":  "padavine_mm",
    "viÅ¡ina_sneÅ¾ne_odeje_cm": "snezna_odeja_cm",
    "viÅ¡ina_novega_snega_cm":  "novi_sneg_cm",
    "toÄ\x8da":                 "toca",
}


@dataclass
class Postaja:
    rank: int
    id: int
    name: str
    dist_km: float

    def __repr__(self):
        return f"#{self.rank} {self.name} (id={self.id}, {self.dist_km:.2f} km)"


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@lru_cache(maxsize=1)
def _load_odseki() -> pd.DataFrame:
    """Load all odseki CSVs, keep only odsek + geometry columns. Cached."""
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
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=1)
def _load_lokacije() -> pd.DataFrame:
    """Load station locations. Cached."""
    return pd.read_csv(LOKACIJE_PATH, encoding="utf-8")


def _odsek_centroid_wgs84(odsek_id: str) -> tuple[float, float]:
    """Return (lon, lat) in WGS84 for the centroid of the given odsek."""
    df = _load_odseki()
    rows = df[df["odsek"] == odsek_id]
    if rows.empty:
        raise KeyError(f"Odsek '{odsek_id}' not found.")
    geom_str = rows.iloc[0]["geometry"]
    geom = wkt.loads(geom_str)
    cx, cy = geom.centroid.x, geom.centroid.y
    lon, lat = _TRANSFORMER.transform(cx, cy)
    return lon, lat


def get_postaje(odsek_id: str) -> dict[str, list[Postaja]]:
    """
    Return top 3 nearest stations for a given odsek ID.

    Args:
        odsek_id: odsek code as string, e.g. "31001"

    Returns:
        {
            "temp": [Postaja, Postaja, Postaja],  # type 2+3 (temperature)
            "all":  [Postaja, Postaja, Postaja],  # type 1+2+3 (all data)
        }

    Raises:
        KeyError: if odsek_id not found
    """
    lon0, lat0 = _odsek_centroid_wgs84(odsek_id)
    stations = _load_lokacije()

    distances = stations.apply(
        lambda r: _haversine_km(lon0, lat0, r["Longitude"], r["Latitude"]),
        axis=1,
    )
    stations = stations.copy()
    stations["dist_km"] = distances

    def top3(type_filter) -> list[Postaja]:
        subset = stations[stations["Type"].isin(type_filter)].nsmallest(3, "dist_km")
        result = []
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            result.append(Postaja(
                rank=rank,
                id=int(row["ID"]),
                name=row["Name"],
                dist_km=round(float(row["dist_km"]), 3),
            ))
        return result

    return {
        "temp": top3([2, 3]),
        "all":  top3([1, 2, 3]),
    }


@lru_cache(maxsize=1)
def _load_vreme() -> pd.DataFrame:
    """Load vreme.csv into a DataFrame indexed by (station_id, datum). Cached."""
    df = pd.read_csv(VREME_PATH, encoding="utf-8", dtype={"station_id": str})
    df = df.rename(columns=_COL_RENAME)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])
    df = df.set_index(["station_id", "datum"]).sort_index()
    return df


def get_vreme(
    odsek_id: str,
    start: str | date,
    end: str | date,
) -> dict[str, pd.DataFrame]:
    """
    Return weather data for the nearest temperature station and the nearest
    all-data station for a given odsek and date range.

    Args:
        odsek_id:  odsek code, e.g. "31001"
        start:     start date inclusive, e.g. "2024-01-01" or date(2024, 1, 1)
        end:       end date inclusive, e.g. "2024-01-31"

    Returns:
        {
            "temp": pd.DataFrame,  # data from nearest type-2/3 station
            "all":  pd.DataFrame,  # data from nearest type-1/2/3 station
        }

    Raises:
        KeyError: if odsek_id not found
    """
    postaje = get_postaje(odsek_id)
    vreme   = _load_vreme()

    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    def fetch(postaja: Postaja) -> pd.DataFrame:
        sid = str(postaja.id)
        if sid not in vreme.index.get_level_values("station_id"):
            return pd.DataFrame()
        df = vreme.loc[sid]
        mask = (df.index >= start_ts) & (df.index <= end_ts)
        df = df.loc[mask].reset_index().rename(columns={"datum": "datum"})
        df.insert(0, "station_id",   postaja.id)
        df.insert(1, "station_name", postaja.name)
        df.insert(2, "dist_km",      postaja.dist_km)
        return df

    return {
        "temp": fetch(postaje["temp"][0]),
        "all":  fetch(postaje["all"][0]),
    }


if __name__ == "__main__":
    import sys

    odsek = sys.argv[1] if len(sys.argv) > 1 else "31001"
    start = sys.argv[2] if len(sys.argv) > 2 else "2024-06-01"
    end   = sys.argv[3] if len(sys.argv) > 3 else "2024-06-07"

    result = get_postaje(odsek)
    print(f"\nOdsek: {odsek}")
    print("\nTop 3 postaje za TEMPERATURO (tip 2+3):")
    for p in result["temp"]:
        print(f"  {p}")
    print("\nTop 3 postaje za VSE PODATKE (tip 1+2+3):")
    for p in result["all"]:
        print(f"  {p}")

    print(f"\n--- Vreme {start} → {end} ---")
    vreme = get_vreme(odsek, start, end)
    print(f"\nTemperatura ({vreme['temp']['station_name'].iloc[0] if not vreme['temp'].empty else 'ni podatkov'}):")
    print(vreme["temp"].to_string(index=False) if not vreme["temp"].empty else "  ni podatkov")
    print(f"\nVsi podatki ({vreme['all']['station_name'].iloc[0] if not vreme['all'].empty else 'ni podatkov'}):")
    print(vreme["all"].to_string(index=False) if not vreme["all"].empty else "  ni podatkov")