"""
Agregacija dnevnih vremenskih meritev ARSO po mesecih.

Vhod:  data/raw/ARSO/vreme.csv
Izhod: data/processed/vreme_mesecno.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parents[2] / "data" / "raw" / "ARSO" / "vreme.csv"
OUT = Path(__file__).parents[2] / "data" / "processed" / "vreme_mesecno.csv"

DATE_FROM = "2000-01-01"  # vključno, None = brez omejitve
DATE_TO = None            # vključno, None = brez omejitve


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Normalizacija imen stolpcev — odstranimo encoding garbage
    df.columns = [
        "station_id", "datum",
        "povp_T", "max_T", "min_T",
        "padavine_mm", "snezna_odeja_cm", "novi_sneg_cm",
        "nevihta", "toca", "viharni_veter",
    ]

    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])

    if DATE_FROM:
        df = df[df["datum"] >= pd.Timestamp(DATE_FROM)]
    if DATE_TO:
        df = df[df["datum"] <= pd.Timestamp(DATE_TO)]

    # Bool stolpci: da -> 1, vse ostalo -> 0
    for col in ["nevihta", "toca", "viharni_veter"]:
        df[col] = (df[col].str.strip().str.lower() == "da").astype(int)

    # Numerični stolpci
    num_cols = ["povp_T", "max_T", "min_T", "padavine_mm", "snezna_odeja_cm", "novi_sneg_cm"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df["leto"] = df["datum"].dt.year
    df["mesec"] = df["datum"].dt.month
    df["leto_mesec"] = df["datum"].dt.to_period("M")

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["station_id", "leto_mesec"])

    agg = grp.agg(
        # Temperatura
        povp_T_avg=("povp_T", "mean"),          # povprečje povprečnih dnevnih T
        max_T_mesec=("max_T", "max"),            # absolutni max v mesecu
        min_T_mesec=("min_T", "min"),            # absolutni min v mesecu
        povp_max_T=("max_T", "mean"),            # povprečje dnevnih maksimumov
        povp_min_T=("min_T", "mean"),            # povprečje dnevnih minimumov
        temp_razpon_avg=("max_T", lambda x: (x - df.loc[x.index, "min_T"]).mean()),  # povp. dnevni razpon

        # Padavine
        padavine_skupaj_mm=("padavine_mm", "sum"),   # skupne padavine v mesecu
        padavine_avg_mm=("padavine_mm", "mean"),      # povprečne dnevne padavine
        dni_s_padavinami=("padavine_mm", lambda x: (x > 0).sum()),  # število dni s padavinami

        # Sneg
        snezna_odeja_max_cm=("snezna_odeja_cm", "max"),
        snezna_odeja_avg_cm=("snezna_odeja_cm", "mean"),
        novi_sneg_skupaj_cm=("novi_sneg_cm", "sum"),
        dni_s_snegom=("novi_sneg_cm", lambda x: (x > 0).sum()),

        # Bool — število dni z dogodkom
        dni_nevihta=("nevihta", "sum"),
        dni_toca=("toca", "sum"),
        dni_viharni_veter=("viharni_veter", "sum"),

        # Meta
        st_dni=("datum", "count"),              # število dni z meritvami v mesecu
    ).reset_index()

    agg["leto"] = agg["leto_mesec"].dt.year
    agg["mesec"] = agg["leto_mesec"].dt.month
    agg["leto_mesec"] = agg["leto_mesec"].astype(str)

    return agg


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Doda delta vrednosti glede na prejšnji mesec (za isto postajo)."""
    df = df.sort_values(["station_id", "leto_mesec"]).reset_index(drop=True)

    delta_cols = {
        "delta_povp_T": "povp_T_avg",
        "delta_padavine": "padavine_skupaj_mm",
        "delta_max_T": "max_T_mesec",
        "delta_min_T": "min_T_mesec",
        "delta_snezna_odeja_max": "snezna_odeja_max_cm",
    }

    for delta_name, src_col in delta_cols.items():
        df[delta_name] = df.groupby("station_id")[src_col].diff()

    return df


def main():
    print(f"Berem: {RAW}")
    df = load_raw(RAW)
    print(f"  Vrstice: {len(df):,}  |  Postaje: {df['station_id'].nunique()}  |  Obdobje: {df['datum'].min().date()} – {df['datum'].max().date()}")

    print("Agregiram po mesecih...")
    monthly = aggregate_monthly(df)

    print("Dodajam delta vrednosti...")
    monthly = add_deltas(monthly)

    # Uredimo stolpce
    col_order = [
        "station_id", "leto_mesec", "leto", "mesec", "st_dni",
        "povp_T_avg", "max_T_mesec", "min_T_mesec", "povp_max_T", "povp_min_T", "temp_razpon_avg",
        "delta_povp_T", "delta_max_T", "delta_min_T",
        "padavine_skupaj_mm", "padavine_avg_mm", "dni_s_padavinami", "delta_padavine",
        "snezna_odeja_max_cm", "snezna_odeja_avg_cm", "novi_sneg_skupaj_cm", "dni_s_snegom", "delta_snezna_odeja_max",
        "dni_nevihta", "dni_toca", "dni_viharni_veter",
    ]
    monthly = monthly[col_order]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUT, index=False, float_format="%.3f")
    print(f"Shranjeno: {OUT}")
    print(f"  Vrstice: {len(monthly):,}  |  Stolpci: {len(monthly.columns)}")
    print(monthly.head(3).to_string())


if __name__ == "__main__":
    main()
