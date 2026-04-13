"""
agg_posek_vreme.py
------------------
Za vsak [ggo, posek] iz posek_processed.csv pripiše agregirane vremenske značilke
iz najbližje ARSO postaje za zadnje leto (12 mesecev do vključno meseca poseka).

Vhod:
  - data/processed/posek_processed.csv
      Vsebuje vse poseke z atributi + stolpci leto in mesec poseka.
  - data/processed/najblizji_odseki_postaje.csv
      Za vsak [ggo, odsek_id] najbližja postaja tipa 123 (padavine) in tipa 23
      (temp + padavine) za vsako leto med 2006 in 2026.
  - data/processed/vreme_mesecno.csv
      Mesečni vremenski povzetki po postajah.

Izhod:
  - data/processed/posek_z_vremenom.csv
      Vse vrstice iz posek_processed.csv + pripisane 12-mesečne agregirane
      vremenske značilke (predpona "leto_").
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
POSEK_IN    = ROOT / "data" / "processed" / "posek_processed.csv"
POSTAJE_IN  = ROOT / "data" / "processed" / "najblizji_odseki_postaje.csv"
VREME_IN    = ROOT / "data" / "processed" / "vreme_mesecno.csv"
OUT_PATH    = ROOT / "data" / "processed" / "agg_posek_meritve.csv"

YEARS = list(range(2006, 2027))

# Vremenske spremenljivke za agregacijo
AGG_COLS = {
    "povp_T_avg":           "mean",
    "max_T_mesec":          "max",
    "min_T_mesec":          "min",
    "padavine_skupaj_mm":   "sum",
    "padavine_avg_mm":      "mean",
    "dni_s_padavinami":     "sum",
    "snezna_odeja_max_cm":  "max",
    "novi_sneg_skupaj_cm":  "sum",
    "dni_s_snegom":         "sum",
    "dni_nevihta":          "sum",
    "dni_toca":             "sum",
    "dni_viharni_veter":    "sum",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_postaje_long(path: Path) -> pd.DataFrame:
    """Naloži najblizji_odseki_postaje in vrni dolgo obliko: ggo | odsek_id | year | station_23 | station_123."""
    df = pd.read_csv(path, low_memory=False)
    df["ggo"] = df["ggo"].astype(str)
    df["odsek_id"] = df["odsek_id"].astype(str)

    rows = []
    for year in YEARS:
        col_23  = f"station_23_{year}"
        col_123 = f"station_123_{year}"
        tmp = df[["ggo", "odsek_id"]].copy()
        tmp["year"] = year
        tmp["station_23"]  = df[col_23].astype(str)  if col_23  in df.columns else np.nan
        tmp["station_123"] = df[col_123].astype(str) if col_123 in df.columns else np.nan
        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    long["station_23"]  = long["station_23"].replace("nan",  np.nan)
    long["station_123"] = long["station_123"].replace("nan", np.nan)
    return long


def precompute_rolling_features(vreme_df: pd.DataFrame) -> pd.DataFrame:
    """
    Za vsako (station_id, leto_mesec) izračunaj agregirane vrednosti za
    zadnjih 12 mesecev (drseče okno 12, min_periods=1).
    """
    vreme_df = vreme_df.copy()
    vreme_df["period"] = pd.to_datetime(vreme_df["leto_mesec"] + "-01").dt.to_period("M")

    result_parts = []
    stations = vreme_df["station_id"].unique()
    print(f"  Predizračun drsečih 12-mes. oken za {len(stations)} postaj...")

    for sid in stations:
        grp = (
            vreme_df[vreme_df["station_id"] == sid]
            .sort_values("period")
            .copy()
        )
        for col in AGG_COLS:
            if col not in grp.columns:
                grp[col] = np.nan
            else:
                grp[col] = pd.to_numeric(grp[col], errors="coerce")

        feat_rows = {
            "station_id": grp["station_id"].values,
            "leto_mesec": grp["leto_mesec"].values,
        }

        for col, func in AGG_COLS.items():
            s = grp[col]
            if func == "sum":
                rolled = s.rolling(2, min_periods=1).sum()
            elif func == "mean":
                rolled = s.rolling(2, min_periods=1).mean()
            elif func == "max":
                rolled = s.rolling(2, min_periods=1).max()
            elif func == "min":
                rolled = s.rolling(2, min_periods=1).min()
            else:
                rolled = s.rolling(2, min_periods=1).mean()
            feat_rows[f"leto_{col}"] = rolled.values

        feat_rows["leto_st_mesecev"] = (
            grp[list(AGG_COLS.keys())[0]]
            .notna()
            .rolling(2, min_periods=1)
            .sum()
            .values
        )

        result_parts.append(pd.DataFrame(feat_rows))

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Naloži poseke
    print(f"Berem poseke: {POSEK_IN}")
    posek = pd.read_csv(POSEK_IN, low_memory=False)
    posek["ggo"]        = posek["ggo"].astype(str).str.strip()
    posek["odsek"]      = posek["odsek"].astype(str).str.strip()
    posek["leto_mesec"] = posek["leto_mesec"].astype(str).str.strip()
    if "leto" not in posek.columns:
        posek["leto"] = posek["leto_mesec"].str[:4].astype(int)
    else:
        posek["leto"] = posek["leto"].astype(int)
    print(
        f"  Poseki: {len(posek):,}  |  "
        f"Unikatnih [ggo, odsek]: {posek[['ggo', 'odsek']].drop_duplicates().shape[0]:,}"
    )

    # 2. Naloži postaje (dolga oblika)
    print(f"\nBerem postaje: {POSTAJE_IN}")
    postaje_long = load_postaje_long(POSTAJE_IN)

    # 3. Naloži vreme in predizračunaj drseče značilke
    print(f"\nBerem vreme: {VREME_IN}")
    vreme_df = pd.read_csv(VREME_IN, low_memory=False)
    vreme_df.columns = vreme_df.columns.str.strip()
    vreme_df["station_id"] = vreme_df["station_id"].astype(str)
    print(f"  Vrstice: {len(vreme_df):,}  |  Postaje: {vreme_df['station_id'].nunique()}")

    print("\nPredizračun drsečih 12-mesečnih značilk po postajah...")
    station_feats = precompute_rolling_features(vreme_df)
    available_pairs = set(zip(station_feats["station_id"], station_feats["leto_mesec"]))
    feat_cols = [c for c in station_feats.columns if c not in ("station_id", "leto_mesec")]
    station_feats_idx = station_feats.set_index(["station_id", "leto_mesec"])
    print(f"  Predizračunane vrstice: {len(station_feats_idx):,}")

    # 4. Za vsak posek poišči pravo postajo (leto poseka → station_23 ali station_123)
    #    Spoji posek z postaje_long na (ggo, odsek == odsek_id, leto == year)
    print("\nIščem postaje za poseke...")
    posek["_year"] = posek["leto"].clip(min(YEARS), max(YEARS))

    merged = posek.merge(
        postaje_long.rename(columns={"odsek_id": "odsek", "year": "_year"}),
        on=["ggo", "odsek", "_year"],
        how="left",
    )
    print(f"  Poseki z najdeno postajo v tabeli: {merged['station_23'].notna().sum():,} / {len(merged):,}")

    # Izberi postajo: prednost station_23 (temp + padavine), rezerva station_123
    def _pick_station(df):
        s23_avail = df["station_23"].notna() & pd.Series(
            [(str(s), m) in available_pairs for s, m in zip(df["station_23"].fillna(""), df["leto_mesec"])],
            index=df.index,
        )
        s123_avail = df["station_123"].notna() & pd.Series(
            [(str(s), m) in available_pairs for s, m in zip(df["station_123"].fillna(""), df["leto_mesec"])],
            index=df.index,
        )
        result = pd.Series(index=df.index, dtype=object)
        result[s23_avail] = df.loc[s23_avail, "station_23"].astype(str)
        result[s123_avail & ~s23_avail] = df.loc[s123_avail & ~s23_avail, "station_123"].astype(str)
        return result

    merged["used_station"] = _pick_station(merged)
    n_mapped = merged["used_station"].notna().sum()
    print(f"  Poseki z najdeno vremensko postajo: {n_mapped:,} / {len(merged):,} "
          f"({100 * n_mapped / len(merged):.1f}%)")

    # 5. Pripiši vremenske značilke
    print("\nPridružujem vremenske značilke...")
    merged["_key"] = list(zip(merged["used_station"].fillna(""), merged["leto_mesec"]))

    has_station = merged["used_station"].notna()
    keys_needed = set(merged.loc[has_station, "_key"])

    feat_subset = station_feats_idx.loc[
        station_feats_idx.index.isin(keys_needed)
    ].reset_index()
    feat_subset["_key"] = list(zip(feat_subset["station_id"], feat_subset["leto_mesec"]))
    key_to_feats = feat_subset.set_index("_key")[feat_cols]

    found = merged["_key"].isin(key_to_feats.index)
    feat_values = key_to_feats.loc[merged.loc[found, "_key"]]
    feat_values.index = merged.loc[found].index

    # Sestavi izhod: samo ključ [ggo, odsek_id, leto_mesec] + used_station + vremenske značilke
    output = merged[["ggo", "odsek", "leto_mesec", "used_station"]].copy()
    output = output.rename(columns={"odsek": "odsek_id"})
    for col in feat_cols:
        output[col] = np.nan
        output.loc[found, col] = feat_values[col].values

    # 6. Shrani
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_PATH, index=False, float_format="%.3f")
    print(f"\nShranjeno: {OUT_PATH}")
    print(f"  Vrstice: {len(output):,}  |  Stolpci: {len(output.columns)}")
    print(f"  Poseki z vremenskimi podatki: {output['used_station'].notna().sum():,}")

    feat_ok = output[feat_cols[0]].notna().sum()
    print(f"  Poseki z izpolnjeno prvo vremensko značilko: {feat_ok:,}")

    print("\nPrimer (prve 3 vrstice):")
    print(output.head(3).to_string())
    print("\nStolpci:")
    print(list(output.columns))


def aggregate() -> "pl.DataFrame":
    """
    Run the full posek-weather aggregation and return result as polars DataFrame.

    Reads from:
      - data/processed/posek_processed.csv
      - data/processed/najblizji_odseki_postaje.csv
      - data/processed/vreme_mesecno.csv

    Returns:
        polars DataFrame with columns: odsek_id, leto_mesec, used_station,
        and rolling 12-month weather features.
    """
    import polars as pl

    main()
    return pl.read_csv(OUT_PATH)


if __name__ == "__main__":
    main()
