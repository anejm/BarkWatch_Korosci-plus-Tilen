"""
agg_posek_meritve.py
--------------------
Za vsak odsek in mesec izračuna agregirane vremenske značilke iz najbližje
ARSO postaje za zadnje leto (12 mesecev vključno s tekočim).

Vhod:
  - data/processed/najblizji_odseki_postaje.csv
      Vsebuje za vsak odsek_id najbližjo postajo tipa 123 (padavine) in
      tipa 23 (temp + padavine) za vsako leto med 2006 in 2026.
  - data/processed/vreme_mesecno.csv
      Mesečni vremenski povzetki po postajah (padavine, temperatura, sneg,
      extremni dogodki).

Izhod:
  - data/processed/agg_posek_meritve.csv
      Ključ: [odsek_id, leto_mesec]
      Stolpci: agregirane vrednosti za zadnje 12 mesecev pred tekočim mescem
               (okno: t-11 … t, vključujoč tekoči mesec)

Strategija postaj:
  - Za vsak odsek in leto se uporabi postaja tipa 23 (ima temperaturo +
    padavine). Postaja tipa 123 (samo padavine) se ne združuje posebej –
    temperatura bo NaN, če postaja 23 ni razpoložljiva za dano postajo/mesec.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
POSTAJE_IN    = ROOT / "data" / "processed" / "najblizji_odseki_postaje.csv"
VREME_IN      = ROOT / "data" / "processed" / "vreme_mesecno.csv"
OUT_PATH      = ROOT / "data" / "processed" / "agg_posek_meritve.csv"

# Years for which year-specific station columns exist
YEARS = list(range(2006, 2027))

# Weather columns to aggregate from vreme_mesecno
# (temp, precipitation, snow, wind/events)
AGG_COLS = {
    # Temperature (type-23 stations only)
    "povp_T_avg":       "mean",   # mean of monthly avg temp  → avg over year
    "max_T_mesec":      "max",    # monthly max temp          → yearly max
    "min_T_mesec":      "min",    # monthly min temp          → yearly min
    # Precipitation
    "padavine_skupaj_mm":  "sum",  # total rainfall mm
    "padavine_avg_mm":     "mean", # avg daily rainfall mm
    "dni_s_padavinami":    "sum",  # rain days count
    # Snow
    "snezna_odeja_max_cm": "max",  # max snow cover cm
    "novi_sneg_skupaj_cm": "sum",  # total new snow cm
    "dni_s_snegom":        "sum",  # snow days count
    # Extreme events
    "dni_nevihta":      "sum",     # storm days
    "dni_toca":         "sum",     # hail days
    "dni_viharni_veter":"sum",     # high-wind days
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(iterable, desc: str = "", total: int | None = None):
    n = total or (len(iterable) if hasattr(iterable, "__len__") else None)
    step = max(1, (n // 20)) if n else 500
    for i, item in enumerate(iterable):
        if i % step == 0:
            print(f"  {desc}: {i}/{n or '?'}", flush=True)
        yield item


def load_postaje(path: Path) -> pd.DataFrame:
    """Load najblizji_odseki_postaje and keep odsek_id + year-specific station_23 cols."""
    df = pd.read_csv(path, low_memory=False)
    df["odsek_id"] = df["odsek_id"].astype(str)

    # Build long table: odsek_id | year | station_23 | station_123
    rows = []
    for year in YEARS:
        col_23  = f"station_23_{year}"
        col_123 = f"station_123_{year}"
        tmp = df[["odsek_id"]].copy()
        tmp["year"] = year
        tmp["station_23"]  = df[col_23].astype(str)  if col_23  in df.columns else np.nan
        tmp["station_123"] = df[col_123].astype(str) if col_123 in df.columns else np.nan
        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)
    # Replace "nan" strings with NaN
    long["station_23"]  = long["station_23"].replace("nan",  np.nan)
    long["station_123"] = long["station_123"].replace("nan", np.nan)
    return long


def precompute_rolling_features(vreme_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (station_id, leto_mesec), aggregate weather columns over the
    last 12 months (rolling window of 12, min_periods=1).

    Returns a DataFrame indexed by (station_id, leto_mesec) with prefixed
    feature columns.
    """
    vreme_df = vreme_df.copy()
    vreme_df["period"] = pd.to_datetime(vreme_df["leto_mesec"] + "-01").dt.to_period("M")

    result_parts = []
    stations = vreme_df["station_id"].unique()
    print(f"  Izračun drsečega letnega okna za {len(stations)} postaj...")

    for sid in _progress(stations, desc="Postaje", total=len(stations)):
        grp = (
            vreme_df[vreme_df["station_id"] == sid]
            .sort_values("period")
            .copy()
        )
        # Ensure all expected columns exist (fill with NaN if absent)
        for col in AGG_COLS:
            if col not in grp.columns:
                grp[col] = np.nan
            else:
                grp[col] = pd.to_numeric(grp[col], errors="coerce")

        feat_rows = {"station_id": grp["station_id"].values,
                     "leto_mesec": grp["leto_mesec"].values}

        for col, func in AGG_COLS.items():
            s = grp[col]
            if func == "sum":
                rolled = s.rolling(12, min_periods=1).sum()
            elif func == "mean":
                rolled = s.rolling(12, min_periods=1).mean()
            elif func == "max":
                rolled = s.rolling(12, min_periods=1).max()
            elif func == "min":
                rolled = s.rolling(12, min_periods=1).min()
            else:
                rolled = s.rolling(12, min_periods=1).mean()
            feat_rows[f"leto_{col}"] = rolled.values

        # Count how many months went into the window
        feat_rows["leto_st_mesecev"] = (
            grp[list(AGG_COLS.keys())[0]]
            .notna()
            .rolling(12, min_periods=1)
            .sum()
            .values
        )

        result_parts.append(pd.DataFrame(feat_rows))

    result = pd.concat(result_parts, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load inputs
    print(f"Berem postaje: {POSTAJE_IN}")
    postaje_long = load_postaje(POSTAJE_IN)
    print(f"  Odseki: {postaje_long['odsek_id'].nunique():,}  |  "
          f"Odsek×leto vrstic: {len(postaje_long):,}")

    print(f"\nBerem vreme: {VREME_IN}")
    vreme_df = pd.read_csv(VREME_IN, low_memory=False)
    vreme_df.columns = vreme_df.columns.str.strip()
    vreme_df["station_id"] = vreme_df["station_id"].astype(str)
    print(f"  Vrstice: {len(vreme_df):,}  |  "
          f"Postaje: {vreme_df['station_id'].nunique()}  |  "
          f"Meseci: {vreme_df['leto_mesec'].nunique()}")

    # 2. Pre-compute rolling 12-month features per (station_id, leto_mesec)
    print("\nPredizračun drsečih 12-mesečnih značilk...")
    station_feats = precompute_rolling_features(vreme_df)
    # Create lookup dict: (station_id, leto_mesec) → feature row
    station_feats_idx = station_feats.set_index(["station_id", "leto_mesec"])
    feat_cols = [c for c in station_feats.columns if c not in ("station_id", "leto_mesec")]
    print(f"  Predizračunane vrstice: {len(station_feats_idx):,}")

    # 3. Build output: for each (odsek_id, leto_mesec) find correct station and fetch features
    #    Use station_23 (temp + rain) as primary; fall back to station_123 (rain only).
    all_months = sorted(vreme_df["leto_mesec"].unique())
    print(f"\nGradiram izhodni nabor za {postaje_long['odsek_id'].nunique():,} odseki "
          f"× {len(all_months)} mesecev...")

    # Expand odseki × months via merge
    months_df = pd.DataFrame({"leto_mesec": all_months})
    months_df["year"] = months_df["leto_mesec"].str[:4].astype(int)

    # Clamp year to YEARS range
    months_df["year"] = months_df["year"].clip(min(YEARS), max(YEARS))

    # Cross-join: odsek_id × months (via year key)
    merged = postaje_long.merge(months_df, on="year", how="inner")
    print(f"  Skupaj parov odsek×mesec: {len(merged):,}")

    # 4. Determine which station to use per row
    #    Prefer station_23 (has temp+rain); fall back to station_123
    available_pairs = set(zip(station_feats["station_id"], station_feats["leto_mesec"]))

    def _pick_station(row):
        s23  = row["station_23"]
        s123 = row["station_123"]
        lm   = row["leto_mesec"]
        if pd.notna(s23) and (str(s23), lm) in available_pairs:
            return str(s23)
        if pd.notna(s123) and (str(s123), lm) in available_pairs:
            return str(s123)
        return None

    print("  Izbiram postaje za vsak par (odsek, mesec)...")
    # Vectorised lookup: try station_23 first, then station_123
    def _vectorised_pick(df):
        s23_avail  = df["station_23"].notna() & pd.Series(
            [(str(s), m) in available_pairs for s, m in zip(df["station_23"], df["leto_mesec"])],
            index=df.index
        )
        s123_avail = df["station_123"].notna() & pd.Series(
            [(str(s), m) in available_pairs for s, m in zip(df["station_123"], df["leto_mesec"])],
            index=df.index
        )
        result = pd.Series(index=df.index, dtype=object)
        result[s23_avail]  = df.loc[s23_avail, "station_23"].astype(str)
        result[s123_avail & ~s23_avail] = df.loc[s123_avail & ~s23_avail, "station_123"].astype(str)
        return result

    merged["used_station"] = _vectorised_pick(merged)

    n_mapped = merged["used_station"].notna().sum()
    print(f"  Parov z najdeno postajo: {n_mapped:,} / {len(merged):,} "
          f"({100*n_mapped/len(merged):.1f}%)")

    # 5. Join features
    print("  Pridružujem vremenske značilke...")
    merged_with_station = merged.dropna(subset=["used_station"]).copy()
    merged_with_station["_key"] = list(zip(
        merged_with_station["used_station"],
        merged_with_station["leto_mesec"]
    ))

    # Batch lookup from index
    feat_df = station_feats_idx.loc[
        station_feats_idx.index.isin(merged_with_station["_key"])
    ].reset_index()
    feat_df["_key"] = list(zip(feat_df["station_id"], feat_df["leto_mesec"]))

    key_to_feats = feat_df.set_index("_key")[feat_cols]

    found_keys = merged_with_station["_key"].isin(key_to_feats.index)
    result_feats = key_to_feats.loc[merged_with_station.loc[found_keys, "_key"]]
    result_feats.index = merged_with_station.loc[found_keys].index

    output = merged_with_station.loc[found_keys, ["odsek_id", "leto_mesec", "used_station"]].copy()
    for col in feat_cols:
        output[col] = result_feats[col].values

    # Add NaN rows for unmatched pairs
    unmatched_with = merged.loc[~merged.index.isin(output.index), ["odsek_id", "leto_mesec"]].copy()
    unmatched_with["used_station"] = np.nan
    for col in feat_cols:
        unmatched_with[col] = np.nan

    output = pd.concat([output, unmatched_with], ignore_index=True)
    output = output.sort_values(["odsek_id", "leto_mesec"]).reset_index(drop=True)

    # 6. Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT_PATH, index=False, float_format="%.3f")
    print(f"\nShranjeno: {OUT_PATH}")
    print(f"  Vrstice: {len(output):,}  |  Stolpci: {len(output.columns)}")
    print("\nPrimer (prve 3 vrstice):")
    print(output.head(3).to_string())
    print("\nStolpci:")
    print(list(output.columns))


if __name__ == "__main__":
    main()
