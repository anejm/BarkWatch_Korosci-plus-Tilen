"""
agg_posek_meritve.py
--------------------
Za vsak posek izračuna agregirane vremenske značilke iz najbližje ARSO postaje.

Časovna okna (pred datumom posekanja):
  - 30d  : zadnjih 30 dni  (kratkoročno)
  - 90d  : zadnjih 90 dni  (sezonsko)
  - 365d : zadnjih 365 dni (letno)

Za vsako okno se izračunajo:
  - temperatura: povp, max, min, dni_mraz (T < 0), dni_vrocina (T > 30)
  - padavine: skupaj_mm, povp_mm, dni_s_padavinami
  - sneg: odeja_max_cm, novi_sneg_skupaj_cm, dni_s_snegom
  - eventi: dni_nevihta, dni_toca, dni_viharni_veter
  - meta: st_dni (koliko dni ima podatke)

Izhod: data/processed/posek_meritve.csv
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from pathlib import Path

def _progress(iterable, desc: str = "", total: int | None = None):
    """Minimal progress printer — no external dependencies."""
    n = total or (len(iterable) if hasattr(iterable, "__len__") else None)
    step = max(1, (n // 20)) if n else 500
    for i, item in enumerate(iterable):
        if i % step == 0:
            print(f"  {desc}: {i}/{n or '?'}", flush=True)
        yield item

sys.path.insert(0, str(Path(__file__).parent))
from bliznje_vremenske_postaje import get_postaje, _load_vreme

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
POSEK_IN  = ROOT / "data" / "raw" / "ZGS" / "posek.csv"
OUT_PATH  = ROOT / "data" / "processed" / "posek_meritve.csv"

# Time windows: label -> days before posekano
WINDOWS = {
    "30d":  30,
    "90d":  90,
    "365d": 365,
}


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _bool_sum(series: pd.Series) -> float:
    """Sum 'da'/'ne' or 0/1 column → count of 'da'/1."""
    s = series.copy()
    if s.dtype == object:
        return (s.str.strip().str.lower() == "da").sum()
    return pd.to_numeric(s, errors="coerce").fillna(0).sum()


def agg_window(df: pd.DataFrame, prefix: str) -> dict:
    """
    Aggregate a daily weather DataFrame for one time window.
    Returns a flat dict with keys prefixed by `prefix`.
    """
    nan_keys = [
        "povp_T", "max_T", "min_T", "dni_mraz", "dni_vrocina",
        "padavine_skupaj_mm", "padavine_avg_mm", "dni_s_padavinami",
        "snezna_odeja_max_cm", "novi_sneg_skupaj_cm", "dni_s_snegom",
        "dni_nevihta", "dni_toca", "dni_viharni_veter",
        "st_dni",
    ]

    if df is None or df.empty:
        return {f"{prefix}_{k}": np.nan for k in nan_keys}

    r = {}

    # --- Temperatura ---
    T = pd.to_numeric(df.get("povp_T_degC", pd.Series(dtype=float)), errors="coerce")
    Tx = pd.to_numeric(df.get("max_T_degC",  pd.Series(dtype=float)), errors="coerce")
    Tn = pd.to_numeric(df.get("min_T_degC",  pd.Series(dtype=float)), errors="coerce")

    r[f"{prefix}_povp_T"]      = T.mean()
    r[f"{prefix}_max_T"]       = Tx.max()
    r[f"{prefix}_min_T"]       = Tn.min()
    r[f"{prefix}_dni_mraz"]    = int((Tn < 0).sum())     if Tn.notna().any() else np.nan
    r[f"{prefix}_dni_vrocina"] = int((Tx > 30).sum())    if Tx.notna().any() else np.nan

    # --- Padavine ---
    pad = pd.to_numeric(df.get("padavine_mm", pd.Series(dtype=float)), errors="coerce")
    r[f"{prefix}_padavine_skupaj_mm"] = pad.sum(min_count=1)
    r[f"{prefix}_padavine_avg_mm"]    = pad.mean()
    r[f"{prefix}_dni_s_padavinami"]   = int((pad > 0).sum()) if pad.notna().any() else np.nan

    # --- Sneg ---
    odeja = pd.to_numeric(df.get("snezna_odeja_cm", pd.Series(dtype=float)), errors="coerce")
    nsneg = pd.to_numeric(df.get("novi_sneg_cm",    pd.Series(dtype=float)), errors="coerce")
    r[f"{prefix}_snezna_odeja_max_cm"]  = odeja.max()
    r[f"{prefix}_novi_sneg_skupaj_cm"]  = nsneg.sum(min_count=1)
    r[f"{prefix}_dni_s_snegom"]         = int((nsneg > 0).sum()) if nsneg.notna().any() else np.nan

    # --- Ekstremni dogodki ---
    for col, feat in [("nevihta", "dni_nevihta"), ("toca", "dni_toca"), ("viharni_veter", "dni_viharni_veter")]:
        if col in df.columns:
            r[f"{prefix}_{feat}"] = int(_bool_sum(df[col]))
        else:
            r[f"{prefix}_{feat}"] = np.nan

    r[f"{prefix}_st_dni"] = len(df)
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load posek
    print(f"Berem posek: {POSEK_IN}")
    df = pd.read_csv(POSEK_IN, low_memory=False)
    df.columns = df.columns.str.strip()
    df["odsek"] = df["odsek"].str.strip()
    df["posekano"] = pd.to_datetime(df["posekano"], errors="coerce")
    df = df.dropna(subset=["odsek", "posekano"]).reset_index(drop=True)
    print(f"  Vrstice: {len(df):,}  |  Odseki: {df['odsek'].nunique()}")

    # 2. Load vreme once (cached)
    print("Nalagam vremenske podatke...")
    vreme_all = _load_vreme()   # MultiIndex (station_id, datum)
    available_stations = set(vreme_all.index.get_level_values("station_id"))
    print(f"  Postaje: {len(available_stations)}  |  "
          f"Obdobje: {vreme_all.index.get_level_values('datum').min().date()} – "
          f"{vreme_all.index.get_level_values('datum').max().date()}")

    # 3. For each unique odsek, resolve nearest "all" station
    print("Iščem najbližje postaje za vsak odsek...")
    unique_odseki = df["odsek"].unique()
    odsek_station: dict[str, str | None] = {}

    for odsek in _progress(unique_odseki, desc="Postaje"):
        try:
            postaje = get_postaje(odsek)
            # Try "all" stations in order, pick first with data in vreme
            sid = None
            for p in postaje["all"]:
                if str(p.id) in available_stations:
                    sid = str(p.id)
                    break
            # Fallback to "temp" stations
            if sid is None:
                for p in postaje["temp"]:
                    if str(p.id) in available_stations:
                        sid = str(p.id)
                        break
            odsek_station[odsek] = sid
        except Exception as e:
            odsek_station[odsek] = None

    n_mapped = sum(1 for v in odsek_station.values() if v is not None)
    print(f"  Odseki z najdeno postajo: {n_mapped}/{len(unique_odseki)}")

    # 4. Compute weather features for each unique (odsek, posekano) pair
    unique_pairs = df[["odsek", "posekano"]].drop_duplicates().reset_index(drop=True)
    print(f"Agregiram vreme za {len(unique_pairs):,} unikatnih parov (odsek × datum)...")

    feature_rows = []
    errors = 0

    for _, row in _progress(unique_pairs.iterrows(), desc="Vreme", total=len(unique_pairs)):
        odsek     = row["odsek"]
        end_date  = row["posekano"]
        sid       = odsek_station.get(odsek)

        feat = {"odsek": odsek, "posekano": end_date}

        if sid is None or sid not in available_stations:
            for win_name in WINDOWS:
                feat.update(agg_window(None, win_name))
            errors += 1
            feature_rows.append(feat)
            continue

        # Slice station data once, then filter per window
        try:
            df_station = vreme_all.loc[sid].reset_index()   # columns: datum + weather cols
            df_station = df_station.rename(columns={"index": "datum"}) if "index" in df_station.columns else df_station
        except Exception:
            for win_name in WINDOWS:
                feat.update(agg_window(None, win_name))
            errors += 1
            feature_rows.append(feat)
            continue

        for win_name, days in WINDOWS.items():
            start_date = end_date - pd.Timedelta(days=days)
            mask = (df_station["datum"] >= start_date) & (df_station["datum"] <= end_date)
            feat.update(agg_window(df_station.loc[mask], win_name))

        feature_rows.append(feat)

    weather_df = pd.DataFrame(feature_rows)
    print(f"  Napake (brez postaje/podatkov): {errors}")

    # 5. Merge back to original posek
    print("Združujem z originalnimi posek podatki...")
    result = df.merge(weather_df, on=["odsek", "posekano"], how="left")

    # 6. Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False, float_format="%.3f")
    print(f"\nShranjeno: {OUT_PATH}")
    print(f"  Vrstice: {len(result):,}  |  Stolpci: {len(result.columns)}")
    print("\nPrimer (prve 3 vrstice):")
    print(result.head(3).to_string())


if __name__ == "__main__":
    main()
