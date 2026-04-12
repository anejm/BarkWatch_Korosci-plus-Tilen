"""
generate_bark_beetle_csv.py
---------------------------
Generates synthetic monthly bark beetle population data for every odsek_id
in posek_processed.csv, covering 2007-01 to 2025-12.

Model (temperature-driven logistic with harvest predation and spatial diffusion):
  1. Temperature suitability: Gaussian response centred on temp_opt
  2. Logistic growth:   growth      = r * B * (1 - B/K)
  3. Temp amplification: temp_growth = beta * temp_effect * B
  4. Harvest predation: predation   = alpha * wood_cut_norm[t-lag] * B
  5. Neighbour diffusion: neighbour_effect = gamma * (mean_neighbour_B - B)
  6. Stochastic noise:  N(0, noise_std)

  B[t] = B[t-1] + growth + temp_growth - predation + neighbour_effect + noise
  B[t] = max(B[t], 0)

Inputs:
  data/processed/posek_processed.csv          – cutting records (odsek, kubikov, leto, mesec)
  data/processed/najblizji_odseki_postaje.csv – nearest weather stations + neighbour odseks
  data/raw/ARSO/vreme.csv                     – daily weather (temp avg/min/max per station)

Output:
  data/synthetic/bark_beetle_by_odsek.csv
    Columns: odsek_id, leto_mesec, bark_beetle_count
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import lil_matrix, diags

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

POSEK_IN   = DATA / "processed" / "posek_processed.csv"
POSTAJE_IN = DATA / "processed" / "najblizji_odseki_postaje.csv"
VREME_IN   = DATA / "raw" / "ARSO" / "vreme.csv"
OUT_PATH   = DATA / "synthetic" / "bark_beetle_by_odsek.csv"

# ---------------------------------------------------------------------------
# Model parameters  (matching the ChatGPT formula in prompt.txt)
# ---------------------------------------------------------------------------
PARAMS = {
    "r":          0.3,    # intrinsic growth rate
    "K":          1000.0, # carrying capacity (beetles per odsek)
    "alpha":      0.002,  # harvest predation strength (applied to normalised wood_cut)
    "beta":       0.15,   # temperature-driven amplification
    "gamma":      0.1,    # neighbour diffusion rate
    "lag":        1,      # months lag before harvest reduces population
    "noise_std":  20.0,   # stochastic noise std
    "temp_opt":   20.0,   # optimal temperature (°C) for bark beetles
    "temp_width": 10.0,   # temperature tolerance (°C, Gaussian half-width)
}

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_monthly_weather(vreme_path: Path, date_idx: dict) -> dict:
    """
    Read vreme.csv, keep 2007–2025, aggregate to monthly avg/min/max per station.
    Returns dict:  {station_id: {"avg": np.array, "max": np.array, "min": np.array}}
    Each array has length T_steps, indexed by date_idx.
    """
    print("  Reading raw weather file …")
    vreme = pd.read_csv(
        vreme_path,
        usecols=[0, 1, 2, 3, 4],
        low_memory=False,
    )
    vreme.columns = ["station_id", "datum", "avg_temp", "max_temp", "min_temp"]
    vreme["datum"] = pd.to_datetime(vreme["datum"], errors="coerce")
    vreme = vreme.dropna(subset=["datum"])
    vreme = vreme[(vreme["datum"].dt.year >= 2007) & (vreme["datum"].dt.year <= 2025)]
    for col in ["avg_temp", "max_temp", "min_temp"]:
        vreme[col] = pd.to_numeric(vreme[col], errors="coerce")

    vreme["leto_mesec"] = vreme["datum"].dt.to_period("M").astype(str)
    monthly = (
        vreme.groupby(["station_id", "leto_mesec"])
        .agg(avg=("avg_temp", "mean"), mx=("max_temp", "mean"), mn=("min_temp", "mean"))
        .reset_index()
    )

    global_avg = float(np.nanmean(monthly["avg"]))
    global_max = float(np.nanmean(monthly["mx"]))
    global_min = float(np.nanmean(monthly["mn"]))
    print(f"  Global mean temp: {global_avg:.2f} °C  |  stations: {monthly['station_id'].nunique()}")

    T = len(date_idx)
    station_weather = {}
    for sid, grp in monthly.groupby("station_id"):
        avg_arr = np.full(T, global_avg, dtype=np.float32)
        max_arr = np.full(T, global_max, dtype=np.float32)
        min_arr = np.full(T, global_min, dtype=np.float32)
        for _, row in grp.iterrows():
            lm = row["leto_mesec"]
            if lm not in date_idx:
                continue
            t = date_idx[lm]
            if pd.notna(row["avg"]):
                avg_arr[t] = float(row["avg"])
            if pd.notna(row["mx"]):
                max_arr[t] = float(row["mx"])
            if pd.notna(row["mn"]):
                min_arr[t] = float(row["mn"])
        station_weather[int(sid)] = {"avg": avg_arr, "max": max_arr, "min": min_arr}

    # Fallback global entry (key -1)
    station_weather[-1] = {
        "avg": np.full(T, global_avg, dtype=np.float32),
        "max": np.full(T, global_max, dtype=np.float32),
        "min": np.full(T, global_min, dtype=np.float32),
    }
    return station_weather, global_avg, global_max, global_min


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    # ── 1. Unique odsek_ids from posek_processed ────────────────────────────
    print("Loading posek data …")
    posek_raw = pd.read_csv(POSEK_IN, low_memory=False)
    posek_raw["kubikov"] = pd.to_numeric(posek_raw["kubikov"], errors="coerce").fillna(0.0)
    posek_raw["odsek_id"] = posek_raw["odsek"].astype(str)

    odseks = sorted(posek_raw["odsek_id"].unique())
    N = len(odseks)
    odsek_idx = {od: i for i, od in enumerate(odseks)}
    print(f"  Unique odseks: {N:,}")

    # Time axis: 2007-01 → 2025-12
    dates     = pd.period_range("2007-01", "2025-12", freq="M")
    T_steps   = len(dates)
    date_strs = [str(p) for p in dates]
    date_idx  = {s: t for t, s in enumerate(date_strs)}
    print(f"  Time steps: {T_steps}")

    # ── 2. Wood cutting matrix  (N × T_steps), normalised per odsek ─────────
    print("Building harvest matrix …")
    agg = (
        posek_raw
        .groupby(["odsek_id", "leto", "mesec"])["kubikov"]
        .sum()
        .reset_index()
    )
    agg["leto_mesec"] = (
        agg["leto"].astype(str) + "-" + agg["mesec"].astype(str).str.zfill(2)
    )
    agg["pi"] = agg["odsek_id"].map(odsek_idx)
    agg["ti"] = agg["leto_mesec"].map(date_idx)
    valid = agg.dropna(subset=["pi", "ti"])

    harvest = np.zeros((N, T_steps), dtype=np.float64)
    harvest[
        valid["pi"].values.astype(int),
        valid["ti"].values.astype(int),
    ] = valid["kubikov"].values

    # Normalise each row (odsek) to [0, 1]
    row_max = harvest.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    harvest_norm = harvest / row_max
    print(f"  Harvest entries mapped: {len(valid):,}")

    # ── 3. Station mapping per odsek ─────────────────────────────────────────
    print("Loading station mapping …")
    postaje = pd.read_csv(POSTAJE_IN, low_memory=False)
    postaje["odsek_id"] = postaje["odsek_id"].astype(str)
    postaje = postaje.set_index("odsek_id")

    # For each odsek pick one station: prefer station_23, fall back to station_123
    parcel_station = np.full(N, -1, dtype=np.int64)
    for od, i in odsek_idx.items():
        if od not in postaje.index:
            continue
        row = postaje.loc[od]
        sid = None
        if "station_23" in postaje.columns and pd.notna(row.get("station_23")):
            sid = int(row["station_23"])
        elif "station_123" in postaje.columns and pd.notna(row.get("station_123")):
            sid = int(row["station_123"])
        if sid is not None:
            parcel_station[i] = sid

    # ── 4. Load and aggregate weather ───────────────────────────────────────
    print("Loading weather data …")
    station_weather, g_avg, g_max, g_min = load_monthly_weather(VREME_IN, date_idx)

    # ── 5. Build temperature matrices  (N × T_steps) ─────────────────────────
    print("Building temperature matrices …")
    avg_T = np.full((N, T_steps), g_avg, dtype=np.float64)
    max_T = np.full((N, T_steps), g_max, dtype=np.float64)
    min_T = np.full((N, T_steps), g_min, dtype=np.float64)

    for i in range(N):
        sid = int(parcel_station[i])
        if sid in station_weather:
            avg_T[i] = station_weather[sid]["avg"]
            max_T[i] = station_weather[sid]["max"]
            min_T[i] = station_weather[sid]["min"]

    # ── 6. Spatial neighbour graph ───────────────────────────────────────────
    print("Building spatial neighbour graph …")
    W = lil_matrix((N, N), dtype=np.float32)
    for od, i in odsek_idx.items():
        if od not in postaje.index:
            continue
        nbrs_str = postaje.loc[od, "bliznji_odseki"]
        if pd.isna(nbrs_str) or not str(nbrs_str).strip():
            continue
        for nid in str(nbrs_str).split(";"):
            nid = nid.strip()
            if nid and nid in odsek_idx:
                j = odsek_idx[nid]
                if i != j:
                    W[i, j] = 1.0

    W = W.tocsr()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W_norm = diags(1.0 / row_sums).dot(W)
    print(f"  Graph edges: {W.nnz:,}")

    # ── 7. Simulation ────────────────────────────────────────────────────────
    p = PARAMS
    K      = p["K"]
    r      = p["r"]
    alpha  = p["alpha"]
    beta   = p["beta"]
    gamma  = p["gamma"]
    lag    = p["lag"]
    t_opt  = p["temp_opt"]
    t_wid  = p["temp_width"]
    noise_std = p["noise_std"]

    # Seed population at ~10 % of K with spatial variation
    B = rng.uniform(0.05, 0.15, size=N) * K

    B_history = np.empty((N, T_steps), dtype=np.float32)

    print(f"\nRunning simulation: {T_steps} months × {N:,} odseks …")
    for t in range(T_steps):

        # --- Temperature suitability (Gaussian) ---
        temp_factor = np.exp(-((avg_T[:, t] - t_opt) ** 2) / (2.0 * t_wid ** 2))
        temp_variability = (max_T[:, t] - min_T[:, t]) / 20.0
        temp_effect = temp_factor * (1.0 + 0.3 * temp_variability)

        # --- Logistic growth ---
        growth = r * B * (1.0 - B / K)

        # --- Temperature-driven amplification ---
        temp_growth = beta * temp_effect * B

        # --- Harvest predation (normalised wood cut, lagged) ---
        lag_t = max(0, t - lag)
        predation = alpha * harvest_norm[:, lag_t] * B

        # --- Neighbour diffusion ---
        if W_norm.nnz > 0:
            mean_nbr_B = np.asarray(W_norm.dot(B)).ravel()
        else:
            mean_nbr_B = B
        neighbour_effect = gamma * (mean_nbr_B - B)

        # --- Stochastic noise ---
        noise = rng.normal(0.0, noise_std, size=N)

        # --- Update & clamp ---
        B = B + growth + temp_growth - predation + neighbour_effect + noise
        B = np.maximum(B, 0.0)

        B_history[:, t] = B.astype(np.float32)

        if t == 0 or (t + 1) % 36 == 0:
            print(
                f"  {date_strs[t]}:  "
                f"mean={B.mean():.1f}  "
                f"median={np.median(B):.1f}  "
                f"max={B.max():.0f}"
            )

    # ── 8. Assemble & save output ─────────────────────────────────────────────
    print("\nAssembling output …")
    result = pd.DataFrame({
        "odsek_id":          np.repeat(odseks, T_steps),
        "leto_mesec":        np.tile(date_strs, N),
        "bark_beetle_count": np.maximum(0, np.round(B_history)).astype(np.int32).ravel(),
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\nSaved → {OUT_PATH}")
    print(f"  Rows:    {len(result):,}")
    print(f"  Columns: {list(result.columns)}")
    print("\nDescriptive stats (bark_beetle_count):")
    print(result["bark_beetle_count"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))
    print("\nSample (first 12 months of first odsek):")
    first = result[result["odsek_id"] == odseks[0]].head(12)
    print(first.to_string(index=False))


if __name__ == "__main__":
    main()
