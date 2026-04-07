"""
Agregacija dnevnih vremenskih meritev ARSO po mesecih — popolna mreža za LSTM.

Za vsako postajo ustvari vrstico za vsak mesec v obdobju DATE_FROM–DATE_TO.
Meseci brez meritev dobijo NaN (razen st_dni=0 in has_data=0).

Vhod:  data/raw/ARSO/vreme.csv
Izhod: data/processed/vreme_mesecno_lstm.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).parents[2] / "data" / "raw" / "ARSO" / "vreme.csv"
OUT = Path(__file__).parents[2] / "data" / "processed" / "vreme_mesecno_lstm.csv"

DATE_FROM = "2000-01-01"
DATE_TO = None            # None = do danes


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

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

    for col in ["nevihta", "toca", "viharni_veter"]:
        df[col] = (df[col].str.strip().str.lower() == "da").astype(int)

    num_cols = ["povp_T", "max_T", "min_T", "padavine_mm", "snezna_odeja_cm", "novi_sneg_cm"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    df["leto_mesec"] = df["datum"].dt.to_period("M")

    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["station_id", "leto_mesec"])

    agg = grp.agg(
        povp_T_avg=("povp_T", "mean"),
        max_T_mesec=("max_T", "max"),
        min_T_mesec=("min_T", "min"),
        povp_max_T=("max_T", "mean"),
        povp_min_T=("min_T", "mean"),
        temp_razpon_avg=("max_T", lambda x: (x - df.loc[x.index, "min_T"]).mean()),
        padavine_skupaj_mm=("padavine_mm", "sum"),
        padavine_avg_mm=("padavine_mm", "mean"),
        dni_s_padavinami=("padavine_mm", lambda x: (x > 0).sum()),
        snezna_odeja_max_cm=("snezna_odeja_cm", "max"),
        snezna_odeja_avg_cm=("snezna_odeja_cm", "mean"),
        novi_sneg_skupaj_cm=("novi_sneg_cm", "sum"),
        dni_s_snegom=("novi_sneg_cm", lambda x: (x > 0).sum()),
        dni_nevihta=("nevihta", "sum"),
        dni_toca=("toca", "sum"),
        dni_viharni_veter=("viharni_veter", "sum"),
        st_dni=("datum", "count"),
    ).reset_index()

    return agg


def build_full_grid(agg: pd.DataFrame) -> pd.DataFrame:
    """Ustvari popolno mrežo station_id × mesec, manjkajoče vrstice zapolni z NaN."""
    date_from = pd.Period(DATE_FROM, freq="M") if DATE_FROM else agg["leto_mesec"].min()
    date_to = pd.Period(DATE_TO, freq="M") if DATE_TO else pd.Period(pd.Timestamp.now(), freq="M")

    all_months = pd.period_range(start=date_from, end=date_to, freq="M")
    all_stations = agg["station_id"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_stations, all_months], names=["station_id", "leto_mesec"]
    )
    full = pd.DataFrame(index=full_index).reset_index()

    merged = full.merge(agg, on=["station_id", "leto_mesec"], how="left")

    # has_data flag: 1 kjer so meritve, 0 kjer ni
    merged["has_data"] = merged["st_dni"].notna().astype(int)
    merged["st_dni"] = merged["st_dni"].fillna(0).astype(int)

    return merged


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Doda delta vrednosti glede na prejšnji mesec. Pri luknjah (has_data=0) je delta NaN."""
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
    agg = aggregate_monthly(df)

    print("Gradim popolno mrežo (station × mesec)...")
    monthly = build_full_grid(agg)

    print("Dodajam delta vrednosti...")
    monthly = add_deltas(monthly)

    monthly["leto"] = monthly["leto_mesec"].dt.year
    monthly["mesec"] = monthly["leto_mesec"].dt.month
    monthly["leto_mesec"] = monthly["leto_mesec"].astype(str)

    col_order = [
        "station_id", "leto_mesec", "leto", "mesec", "has_data", "st_dni",
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

    n_missing = (monthly["has_data"] == 0).sum()
    pct = n_missing / len(monthly) * 100
    print(f"  Vrstice brez meritev: {n_missing:,} ({pct:.1f}%)")
    print(monthly.head(3).to_string())


if __name__ == "__main__":
    main()
