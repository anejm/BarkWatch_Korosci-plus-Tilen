"""
Lookup top 3 weather stations and weather data for a given odsek ID.

Usage:
    from postaje_lookup import get_postaje, get_vreme

    # nearest stations
    result = get_postaje("31001")
    print(result["temp"])   # top 3 temperature stations (type 2+3)
    print(result["all"])    # top 3 all-data stations (type 1+2+3)

    # weather data for date range
    vreme = get_vreme("31001", "2024-01-01", "2024-01-31")
    print(vreme["temp"])    # DataFrame from nearest temp station
    print(vreme["all"])     # DataFrame from nearest all-data station
"""

import csv
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass
from datetime import date

import pandas as pd

CSV_PATH   = Path(__file__).parent / "odseki_postaje.csv"
VREME_PATH = Path(__file__).parent / "vreme.csv"

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
        return f"#{self.rank} {self.name} (id={self.id}, {self.dist_km} km)"


@lru_cache(maxsize=1)
def _load_index() -> dict[str, dict]:
    """Load CSV into dict keyed by odsek code. Cached after first call."""
    index = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            index[row["odsek"]] = row
    return index


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
    index = _load_index()
    if odsek_id not in index:
        raise KeyError(f"Odsek '{odsek_id}' not found. "
                       f"Example valid ID: '{next(iter(index))}'")

    row = index[odsek_id]

    def parse_group(prefix: str) -> list[Postaja]:
        result = []
        for rank in range(1, 4):
            p = f"{prefix}_{rank}"
            result.append(Postaja(
                rank=rank,
                id=int(row[f"{p}_id"]),
                name=row[f"{p}_name"],
                dist_km=float(row[f"{p}_dist_km"]),
            ))
        return result

    return {
        "temp": parse_group("top3_temp"),
        "all":  parse_group("top3_all"),
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
        Each DataFrame has columns: datum, station_id, station_name,
        povp_T_degC, max_T_degC, min_T_degC, padavine_mm,
        snezna_odeja_cm, novi_sneg_cm, nevihta, toca, viharni_veter

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
