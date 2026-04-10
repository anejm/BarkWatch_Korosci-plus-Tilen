"""
Generira CSV datoteko z vsemi ARSO postajami, ki so imele kakršne koli meritve
za vsako leto od 2006 do 2026.

Izhod: data/processed/postaje_po_letih.csv
Stolpci: leto, station_id, tip, Longitude, Latitude, Altitude, n_dni

Tip postaje:
  - iz lokacije.csv, če je podan
  - 3 = postaja je merila temperaturo (povp/max/min)
  - 1 = postaja ni merila temperature

Lokacija:
  - iz lokacije.csv, če je podana
  - sicer iz ARSO locations.xml API (poizvedba za vsako leto)
"""

import os
import re
import random
import string
import time
from pathlib import Path

import pandas as pd
import requests

# --- Poti ---
_search = Path(os.getcwd()).resolve()
ROOT = None
for _ in range(5):
    if (_search / "src").exists() and (_search / "data").exists():
        ROOT = _search
        break
    _search = _search.parent
if ROOT is None:
    ROOT = Path("/home/anejm/Documents/hekaton/BarkWatch-arnes_hackathon2026")

VREME_PATH    = ROOT / "data" / "raw" / "ARSO" / "vreme.csv"
LOKACIJE_PATH = ROOT / "data" / "raw" / "ARSO" / "lokacije.csv"
IZHOD_PATH    = ROOT / "data" / "processed" / "postaje_po_letih.csv"

LETO_OD = 2006
LETO_DO = 2026

LOCATIONS_URL = "https://meteo.arso.gov.si/webmet/archive/locations.xml"
LOC_PATTERN   = re.compile(
    r'_(\d+):\{\s*name:"([^"]+)",\s*lon:([\d.\-]+),\s*lat:([\d.\-]+),\s*alt:([\d.\-]+),\s*type:(\d+)'
)

def nocache():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))

def preberi_lokacije_api(leto):
    """Poizvedba locations.xml za dano leto, vrne dict station_id -> (lon, lat, alt)."""
    params = {
        "d1": f"{leto}-01-01",
        "d2": f"{leto}-12-31",
        "type": "3,2,1",
        "lang": "si",
        "nocache": nocache(),
    }
    try:
        r = requests.get(LOCATIONS_URL, params=params, timeout=20)
        rezultat = {}
        for m in LOC_PATTERN.finditer(r.text):
            sid = m.group(1)
            rezultat[sid] = {
                "Longitude": float(m.group(3)),
                "Latitude":  float(m.group(4)),
                "Altitude":  float(m.group(5)),
            }
        return rezultat
    except Exception as e:
        print(f"  ✗ Napaka pri letu {leto}: {e}")
        return {}

def main():
    # --- Naloži podatke ---
    print("Nalagam vreme.csv ...")
    df = pd.read_csv(VREME_PATH, dtype={"station_id": str})
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])
    df["leto"] = df["datum"].dt.year

    print("Nalagam lokacije.csv ...")
    lokacije = pd.read_csv(LOKACIJE_PATH, dtype={"ID": str})
    lokacije = lokacije.rename(columns={
        "ID": "station_id", "Name": "ime", "Type": "tip",
        "Longitude": "Longitude", "Latitude": "Latitude", "Altitude": "Altitude",
    })

    # --- Filtriraj leta 2006–2026 ---
    df_fil = df[df["leto"].between(LETO_OD, LETO_DO)]

    # --- Postaje z vsaj eno meritvijo po letu ---
    rezultat = (
        df_fil.groupby(["leto", "station_id"])
        .size()
        .rename("n_dni")
        .reset_index()
    )

    # --- Dodaj tip in lokacijo iz lokacije.csv ---
    rezultat = rezultat.merge(
        lokacije[["station_id", "tip", "Longitude", "Latitude", "Altitude"]],
        on="station_id",
        how="left",
    )

    # --- Za postaje brez tipa: določi glede na to ali merijo temperaturo ---
    T_COLS = ["povp_dnevna_T_ÂdegC", "max_T_ÂdegC", "min_T_ÂdegC"]
    meri_temp = (
        df_fil.groupby("station_id")[T_COLS]
        .apply(lambda g: g.notna().any().any())
        .rename("meri_temp")
    )
    rezultat = rezultat.merge(meri_temp, on="station_id", how="left")
    mask_brez_tipa = rezultat["tip"].isna()
    rezultat.loc[mask_brez_tipa, "tip"] = rezultat.loc[mask_brez_tipa, "meri_temp"].map(
        {True: 3, False: 1}
    )
    rezultat = rezultat.drop(columns=["meri_temp"])

    # --- Zapolni manjkajoče lokacije iz ARSO API ---
    postaje_brez = rezultat[rezultat["Longitude"].isna()]["station_id"].unique()
    print(f"\nPostaj brez lokacije v lokacije.csv: {len(postaje_brez)}")
    print("Poizvedujem ARSO locations.xml po letih ...")

    # Zberemo koordinate iz vsakega leta — vzamemo prvo najdeno za vsako postajo
    api_lokacije = {}  # station_id -> {Longitude, Latitude, Altitude}

    for leto in range(LETO_OD, LETO_DO + 1):
        # Poizveduj samo dokler so še postaje brez lokacije
        still_missing = [sid for sid in postaje_brez if sid not in api_lokacije]
        if not still_missing:
            break

        print(f"  {leto} ...", end=" ", flush=True)
        podatki = preberi_lokacije_api(leto)

        found = 0
        for sid in still_missing:
            if sid in podatki:
                api_lokacije[sid] = podatki[sid]
                found += 1
        print(f"novih: {found}/{len(still_missing)}")
        time.sleep(0.5)

    # Zapolni v rezultatu
    for sid, lok in api_lokacije.items():
        mask = (rezultat["station_id"] == sid) & rezultat["Longitude"].isna()
        rezultat.loc[mask, "Longitude"] = lok["Longitude"]
        rezultat.loc[mask, "Latitude"]  = lok["Latitude"]
        rezultat.loc[mask, "Altitude"]  = lok["Altitude"]

    rezultat = rezultat[["leto", "station_id", "tip", "Longitude", "Latitude", "Altitude", "n_dni"]].sort_values(
        ["leto", "station_id"]
    )

    # --- Shrani ---
    IZHOD_PATH.parent.mkdir(parents=True, exist_ok=True)
    rezultat.to_csv(IZHOD_PATH, index=False)

    brez_lok = rezultat[rezultat["Longitude"].isna()]["station_id"].nunique()
    print(f"\nShraneno: {IZHOD_PATH}")
    print(f"Skupaj vrstic: {len(rezultat):,}")
    print(f"Postaj brez lokacije (NaN): {brez_lok}")
    print(f"\nPostaj po letu:")
    print(rezultat.groupby("leto")["station_id"].nunique().to_string())


if __name__ == "__main__":
    main()