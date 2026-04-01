import pandas as pd
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "raw" / "ZGS" / "posek.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "posek_processed.csv"

# --- Load ---
df = pd.read_csv(INPUT_PATH)

# --- Strip whitespace ---
df.columns = df.columns.str.strip()
df["odsek"] = df["odsek"].str.strip()

# --- Select and rename ---
df = df[["odsek", "posekano", "kubikov"]].rename(columns={
    "odsek": "ID_ODSEK",
    "posekano": "date",
    "kubikov": "posek",
})

# --- Parse date and drop missing ---
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["ID_ODSEK", "date", "posek"])

# --- Set index and sort ---
df = df.set_index(["ID_ODSEK", "date"]).sort_index()

# --- Save ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH)
print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
