"""
Generira CSV datoteko z vsemi ARSO postajami, ki so imele kakršne koli meritve
za vsako leto od 2006 do 2026.

Izhod: data/processed/postaje_po_letih.csv
Stolpci: leto, station_id, tip, n_dni

Tip postaje:
  - iz lokacije.csv, če je podan
  - 3 = postaja je merila temperaturo (povp/max/min)
  - 1 = postaja ni merila temperature
"""

import os
from pathlib import Path

import pandas as pd

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

VREME_PATH = ROOT / "data" / "raw" / "ARSO" / "vreme.csv"
LOKACIJE_PATH = ROOT / "data" / "raw" / "ARSO" / "lokacije.csv"
IZHOD_PATH = ROOT / "data" / "processed" / "postaje_po_letih.csv"

LETO_OD = 2006
LETO_DO = 2026

# --- Naloži podatke ---
print("Nalagam vreme.csv ...")
df = pd.read_csv(VREME_PATH, dtype={"station_id": str})
df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
df = df.dropna(subset=["datum"])
df["leto"] = df["datum"].dt.year

print("Nalagam lokacije.csv ...")
lokacije = pd.read_csv(LOKACIJE_PATH, dtype={"ID": str})
lokacije = lokacije.rename(columns={"ID": "station_id", "Name": "ime", "Type": "tip"})

# --- Filtriraj leta 2006–2026 ---
df_fil = df[df["leto"].between(LETO_OD, LETO_DO)]

# --- Postaje z vsaj eno meritvijo po letu ---
rezultat = (
    df_fil.groupby(["leto", "station_id"])
    .size()
    .rename("n_dni")
    .reset_index()
)

# --- Dodaj tip postaje iz lokacije.csv ---
rezultat = rezultat.merge(
    lokacije[["station_id", "tip"]],
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

rezultat = rezultat[["leto", "station_id", "tip", "n_dni"]].sort_values(
    ["leto", "station_id"]
)

# --- Shrani ---
IZHOD_PATH.parent.mkdir(parents=True, exist_ok=True)
rezultat.to_csv(IZHOD_PATH, index=False)

print(f"\nShraneno: {IZHOD_PATH}")
print(f"Skupaj vrstic: {len(rezultat):,}")
print(f"\nPostaj po letu:")
print(rezultat.groupby("leto")["station_id"].nunique().to_string())
