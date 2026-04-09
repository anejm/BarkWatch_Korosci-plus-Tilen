"""
agg_sestoji_odseke.py
---------------------
Združi odseki_processed.csv in sestoji_processed.csv v eno tabelo.

Sestoji so agregirani po 'odsek' s sledečo strategijo:
  - 'povrsina' in vsi stolpci z 'lz' predpono (lesne zaloge, deleži vrst):
      → SUM (skupna vrednost za odsek)
  - Ostali numerični stolpci (pompov, etigl, etlst, etsku):
      → MEAN, SUM, STD, MIN, MAX
  - Binarni / one-hot stolpci (ggo_*, rfaza_*, sklep_*, zasnova_*, negovan_*, pomzas_*):
      → MEAN (delež sestoji z dano lastnostjo)
  - Metapodatek: 'sestoji_count' (število sestoji na odsek)

Vhod:
  - data/processed/odseki_processed.csv   (primarni ključ: odsek)
  - data/processed/sestoji_processed.csv  (tuji ključ: odsek)

Izhod:
  - data/processed/agg_odseki_sestoji.csv (ključ: odsek_id)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
ODSEKI_IN    = ROOT / "data" / "processed" / "odseki_processed.csv"
SESTOJI_IN   = ROOT / "data" / "processed" / "sestoji_processed.csv"
OUT_PATH     = ROOT / "data" / "processed" / "agg_odseki_sestoji.csv"

# ---------------------------------------------------------------------------
# Column classification helpers
# ---------------------------------------------------------------------------

def classify_columns(df: pd.DataFrame, key_cols: list[str]) -> dict[str, list[str]]:
    """
    Split DataFrame columns into groups based on naming and dtype.

    Returns dict with keys:
      lz_sum   – 'lz*' columns + 'povrsina' → aggregate with SUM
      num_full – other numeric columns → SUM + MEAN + STD + MIN + MAX
      bool_    – boolean / one-hot columns → MEAN (proportion)
    """
    cols = [c for c in df.columns if c not in key_cols]

    lz_sum   = []
    num_full = []
    bool_    = []

    for col in cols:
        if col.startswith("lz") or col == "povrsina":
            lz_sum.append(col)
        elif df[col].dtype == bool or (
            df[col].dtype == object and df[col].dropna().isin([True, False, "True", "False"]).all()
        ):
            bool_.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            num_full.append(col)
        # non-numeric, non-bool columns (strings etc.) are skipped

    return {"lz_sum": lz_sum, "num_full": num_full, "bool_": bool_}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_sestoji(sestoji: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sestoji_processed by 'odsek'.
    Returns one row per unique odsek with prefixed columns.
    """
    groups = classify_columns(sestoji, key_cols=["odsek", "sestoj"])

    print(f"  lz/povrsina (sum):     {len(groups['lz_sum'])} cols")
    print(f"  Numeric full agg:      {len(groups['num_full'])} cols")
    print(f"  Boolean/one-hot (mean):{len(groups['bool_'])} cols")

    grp = sestoji.groupby("odsek", sort=False)

    parts = []

    # 1. Count of sestoji per odsek
    count_s = grp.size().rename("sestoji_count").reset_index()
    parts.append(count_s)

    # 2. SUM for lz* + povrsina columns
    if groups["lz_sum"]:
        lz_agg = (
            grp[groups["lz_sum"]]
            .sum(min_count=1)
            .rename(columns={c: f"sestoji_{c}_sum" for c in groups["lz_sum"]})
            .reset_index()
        )
        parts.append(lz_agg)

    # 3. SUM + MEAN + STD + MIN + MAX for other numeric columns
    if groups["num_full"]:
        for fn, label in [("sum", "sum"), ("mean", "mean"), ("std", "std"),
                          ("min", "min"), ("max", "max")]:
            agg = (
                grp[groups["num_full"]]
                .agg(fn)
                .rename(columns={c: f"sestoji_{c}_{label}" for c in groups["num_full"]})
                .reset_index()
            )
            parts.append(agg)

    # 4. MEAN for boolean / one-hot columns (→ proportion of sestoji with that flag)
    if groups["bool_"]:
        # Cast bools to int first for mean
        bool_df = sestoji[["odsek"] + groups["bool_"]].copy()
        for col in groups["bool_"]:
            bool_df[col] = pd.to_numeric(bool_df[col], errors="coerce")
        bool_agg = (
            bool_df.groupby("odsek", sort=False)[groups["bool_"]]
            .mean()
            .rename(columns={c: f"sestoji_{c}_mean" for c in groups["bool_"]})
            .reset_index()
        )
        parts.append(bool_agg)

    # Merge all parts on 'odsek'
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on="odsek", how="left")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load odseki_processed
    print(f"Berem odseki: {ODSEKI_IN}")
    odseki = pd.read_csv(ODSEKI_IN, low_memory=False)
    odseki["odsek"] = odseki["odsek"].astype(str)
    print(f"  Vrstice: {len(odseki):,}  |  Unikatnih odsek: {odseki['odsek'].nunique():,}")

    # Deduplicate: keep first row per odsek (odsek is declared primary key)
    n_before = len(odseki)
    odseki = odseki.drop_duplicates(subset="odsek", keep="first").reset_index(drop=True)
    if len(odseki) < n_before:
        print(f"  Odstranjenih duplikatov: {n_before - len(odseki):,}")

    # 2. Load sestoji_processed
    print(f"\nBerem sestoji: {SESTOJI_IN}")
    sestoji = pd.read_csv(SESTOJI_IN, low_memory=False)
    sestoji["odsek"] = sestoji["odsek"].astype(str)
    print(f"  Vrstice: {len(sestoji):,}  |  Unikatnih odsek: {sestoji['odsek'].nunique():,}")

    # Coerce numeric columns
    skip = {"odsek", "sestoj"}
    for col in sestoji.columns:
        if col in skip:
            continue
        if sestoji[col].dtype == object:
            sestoji[col] = pd.to_numeric(sestoji[col], errors="coerce")

    # 3. Aggregate sestoji
    print("\nAgregiram sestoji po odsek...")
    sestoji_agg = aggregate_sestoji(sestoji)
    print(f"  Rezultat: {len(sestoji_agg):,} vrstic  |  {len(sestoji_agg.columns)} stolpcev")

    # 4. Join odseki + aggregated sestoji
    print("\nZdružujem odseki z agregiranimi sestoji...")
    result = odseki.merge(sestoji_agg, on="odsek", how="left")
    print(f"  Skupaj vrstic: {len(result):,}  |  Stolpcev: {len(result.columns)}")

    # Check coverage
    matched = result["sestoji_count"].notna().sum()
    print(f"  Odseki z vsaj enim sestojem: {matched:,} / {len(result):,} "
          f"({100 * matched / len(result):.1f}%)")

    # 5. Rename 'odsek' → 'odsek_id' for clarity
    result = result.rename(columns={"odsek": "odsek_id"})

    # 6. Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False, float_format="%.4f")
    print(f"\nShranjeno: {OUT_PATH}")
    print(f"  Vrstice: {len(result):,}  |  Stolpci: {len(result.columns)}")
    print("\nPrimer (prva vrstica):")
    print(result.head(1).T.to_string())
    print("\nStolpci:")
    print(list(result.columns))


if __name__ == "__main__":
    main()
