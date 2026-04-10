"""
agg_posek_sosedi.py
-------------------
Za vsak [ggo, odsek] in mesec izračuna agregirane značilke poseka iz sosednjih
odseki (v razdalji 1 km od središča odseka).

Vhod:
  - data/processed/najblizji_odseki_postaje.csv
      Vsebuje za vsak [ggo, odsek_id] seznam bližnjih odseki (bliznji_odseki, ločeni
      s podpičjem).
  - data/processed/posek_processed.csv
      Mesečni posek po odsekih: target, lagi, drsečia povprečja, itd.

Izhod:
  - data/processed/agg_posek_sosedi.csv
      Ključ: [ggo, odsek_id, leto_mesec]
      Stolpci: agregirane vrednosti poseka sosednjih odseki (vsota, povprečje,
               std, ...) za vsak mesec.

Strategija agregacije:
  - Za vsak mesec se za vsak [ggo, odsek] poiščejo sosednji odseki (iz bliznji_odseki)
    in se iz posek_processed pridobijo njihove vrednosti za ta mesec.
  - Iz teh vrednosti se izračunajo: vsota (skupna intenzivnost v okolici),
    povprečje, std, mediana in število sosedov s podatki.
  - Obdelava poteka mesec po mesec (ne vse naenkrat), da se izogne
    prekomerni porabi pomnilnika (~264M vrstic pri polni spojitvi).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
POSTAJE_IN  = ROOT / "data" / "processed" / "najblizji_odseki_postaje.csv"
POSEK_IN    = ROOT / "data" / "processed" / "posek_processed.csv"
OUT_PATH    = ROOT / "data" / "processed" / "agg_posek_sosedi.csv"

# Columns from posek_processed to aggregate over neighbors.
# Only use lagged/rolled features — NOT the contemporaneous 'target' or
# 'log1p_target', which represent the CURRENT month's harvest and would
# not be available at prediction time. Lag_1 corresponds to the previous
# month's neighbour harvest, which is a safe leading indicator.
AGG_SPEC = {
    "lag_1":           ["sum", "mean"],
    "lag_3":           ["sum", "mean"],
    "lag_6":           ["sum", "mean"],
    "lag_12":          ["sum", "mean"],
    "rolling_mean_3":  ["mean"],
    "rolling_mean_6":  ["mean"],
    "rolling_mean_12": ["mean"],
    "rolling_std_12":  ["mean"],
}

POSEK_COLS = ["ggo", "odsek", "leto_mesec", "target"] + list(AGG_SPEC.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _progress(iterable, desc: str = "", total: int | None = None):
    n = total or (len(iterable) if hasattr(iterable, "__len__") else None)
    step = max(1, (n // 20)) if n else 50
    for i, item in enumerate(iterable):
        if i % step == 0:
            print(f"  {desc}: {i}/{n or '?'}", flush=True)
        yield item


def build_edge_list(postaje_path: Path) -> pd.DataFrame:
    """
    Parse bliznji_odseki column and return a long edge table:
      ggo | odsek_id | neighbor_odsek
    """
    df = pd.read_csv(postaje_path, usecols=["ggo", "odsek_id", "bliznji_odseki"],
                     low_memory=False)
    df["ggo"] = df["ggo"].astype(str)
    df["odsek_id"] = df["odsek_id"].astype(str)
    df["bliznji_odseki"] = df["bliznji_odseki"].fillna("")

    # Explode semicolon-separated neighbor lists
    exploded = df["bliznji_odseki"].str.split(";").explode()
    exploded = exploded.str.strip().replace("", np.nan).dropna()

    edges = pd.DataFrame({
        "ggo":      df.loc[exploded.index, "ggo"].values,
        "odsek_id":      df.loc[exploded.index, "odsek_id"].values,
        "neighbor_odsek": exploded.values,
    })
    # Drop self-loops
    edges = edges[edges["odsek_id"] != edges["neighbor_odsek"]].reset_index(drop=True)
    return edges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Build edge list
    print(f"Gradim seznam sosedov: {POSTAJE_IN}")
    edges = build_edge_list(POSTAJE_IN)
    print(f"  Skupaj robov (odsek → sosed): {len(edges):,}")
    print(f"  Unikatnih [ggo, odsek_id] v izhodišču: {edges[['ggo', 'odsek_id']].drop_duplicates().shape[0]:,}")

    # 2. Load posek_processed (only needed columns)
    print(f"\nBerem posek: {POSEK_IN}")
    posek_cols_present = []
    header = pd.read_csv(POSEK_IN, nrows=0).columns.tolist()
    for col in POSEK_COLS:
        if col in header:
            posek_cols_present.append(col)
        else:
            print(f"  OPOZORILO: stolpec '{col}' ni v posek_processed, preskakujem.")

    posek = pd.read_csv(POSEK_IN, usecols=posek_cols_present, low_memory=False)
    posek["ggo"] = posek["ggo"].astype(str)
    posek["odsek"] = posek["odsek"].astype(str)

    numeric_cols = [c for c in posek_cols_present if c not in ("ggo", "odsek", "leto_mesec")]
    for col in numeric_cols:
        posek[col] = pd.to_numeric(posek[col], errors="coerce")

    all_months = sorted(posek["leto_mesec"].unique())
    print(f"  Vrstice: {len(posek):,}  |  Odseki: {posek['odsek'].nunique():,}  "
          f"|  Meseci: {len(all_months)}")

    # 3. Pre-index posek by month for fast access
    posek_by_month = {m: grp.drop(columns="leto_mesec")
                      for m, grp in posek.groupby("leto_mesec")}

    # 4. Build output feature names
    feat_cols = []
    agg_funcs: dict[str, list[str]] = {}
    for col, funcs in AGG_SPEC.items():
        if col not in numeric_cols:
            continue
        for fn in funcs:
            feat_cols.append(f"sosedi_{col}_{fn}")
        agg_funcs[col] = funcs
    feat_cols.append("sosedi_st_sosedov")  # number of neighbors with data

    # 5. Process month by month
    print(f"\nAgregiram {len(all_months)} mesecev × {edges[['ggo', 'odsek_id']].drop_duplicates().shape[0]:,} [ggo, odsek_id]...")

    all_results = []

    for leto_mesec in _progress(all_months, desc="Meseci", total=len(all_months)):
        posek_m = posek_by_month.get(leto_mesec)
        if posek_m is None or posek_m.empty:
            continue

        # Join: edges (ggo, odsek_id, neighbor_odsek) × posek_m (ggo, odsek, ...)
        joined = edges.merge(
            posek_m,
            left_on=["ggo", "neighbor_odsek"],
            right_on=["ggo", "odsek"],
            how="left",
        ).drop(columns="odsek")

        # Aggregate by [ggo, odsek_id]
        grp = joined.groupby(["ggo", "odsek_id"], sort=False)

        agg_parts = []
        for col, funcs in agg_funcs.items():
            if col not in joined.columns:
                continue
            s = grp[col]
            for fn in funcs:
                if fn == "sum":
                    agg_parts.append(s.sum(min_count=1).rename(f"sosedi_{col}_sum"))
                elif fn == "mean":
                    agg_parts.append(s.mean().rename(f"sosedi_{col}_mean"))
                elif fn == "std":
                    agg_parts.append(s.std().rename(f"sosedi_{col}_std"))
                elif fn == "median":
                    agg_parts.append(s.median().rename(f"sosedi_{col}_median"))

        # Count neighbors that have non-null target
        count_s = grp["target"].count().rename("sosedi_st_sosedov")
        agg_parts.append(count_s)

        month_df = pd.concat(agg_parts, axis=1).reset_index()
        month_df.insert(2, "leto_mesec", leto_mesec)
        all_results.append(month_df)

    # 6. Concatenate and save
    print("\nZdružujem rezultate...")
    result = pd.concat(all_results, ignore_index=True)
    result = result.sort_values(["ggo", "odsek_id", "leto_mesec"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False, float_format="%.4f")
    print(f"\nShranjeno: {OUT_PATH}")
    print(f"  Vrstice: {len(result):,}  |  Stolpci: {len(result.columns)}")
    print("\nPrimer (prve 3 vrstice z vrednostmi):")
    sample = result[result["sosedi_st_sosedov"] > 0].head(3)
    print(sample.to_string())
    print("\nStolpci:")
    print(list(result.columns))


def aggregate() -> "pl.DataFrame":
    """
    Run the full posek-neighbor aggregation and return result as polars DataFrame.

    Reads from:
      - data/processed/najblizji_odseki_postaje.csv
      - data/processed/posek_processed.csv

    Returns:
        polars DataFrame with columns: odsek_id, leto_mesec,
        and aggregated neighbor posek features (sosedi_* prefix).
    """
    import polars as pl

    main()
    return pl.read_csv(OUT_PATH)


if __name__ == "__main__":
    main()
