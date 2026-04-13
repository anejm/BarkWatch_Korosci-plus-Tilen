"""
generate_bark_beetle_csv.py
---------------------------
Generates synthetic monthly bark beetle population data for every odsek_id
in posek_processed.csv, covering 2007-01 to 2025-12.

Model (temperature-driven logistic with harvest predation and spatial diffusion):
  1. Temperature suitability: Gaussian response centred on temp_opt
  2. Logistic growth:   growth      = r * B * (1 - B/K)
  3. Temp amplification: temp_growth = beta * temp_effect * B
  4. Harvest predation:  current     = alpha   * harvest_norm[t]   * B
                         forward     = alpha_h * harvest_h1_norm[t] * B
                         (h1 from target.csv gives next-month expected harvest)
  5. Neighbour diffusion: neighbour_effect = gamma * (mean_neighbour_B - B)
  6. Stochastic noise:  N(0, noise_std)

  B[t] = B[t-1] + growth + temp_growth - predation - forward_predation
                + neighbour_effect + noise
  B[t] = max(B[t], 0)

Inputs:
  data/processed/posek_processed.csv          – monthly harvest per odsek
                                                (target column; log1p if LOG_TARGET)
  data/processed/target.csv                   – h1..h12 forward harvest horizons
                                                (log1p if LOG_TARGET; h1 used as
                                                 next-month harvest signal)
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
# Log-space flag — must match LOG_TARGET in posek_processing.py
# ---------------------------------------------------------------------------
LOG_TARGET: bool = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

POSEK_IN   = DATA / "processed" / "posek_processed.csv"
TARGET_IN  = DATA / "processed" / "target.csv"
POSTAJE_IN = DATA / "processed" / "najblizji_odseki_postaje.csv"
VREME_IN   = DATA / "raw" / "ARSO" / "vreme.csv"
OUT_PATH   = DATA / "synthetic" / "bark_beetle_by_odsek.csv"

# ---------------------------------------------------------------------------
# Model parameters
# Output unit: beetles / m²
#
# Target steady-state ranges (calibrated against Slovenian station temperatures):
#   Winter  (Dec–Feb):              10 –   200  beetles / m²
#   Building outbreak (spring/autumn): 200 – 2 000  beetles / m²
#   Peak outbreak (Jun–Sep):       2 000 – 10 000  beetles / m²
# ---------------------------------------------------------------------------
PARAMS = {
    # Logistic growth — gated by average temperature
    "r":           0.80,   # max monthly intrinsic growth rate (summer optimum)
    "K":       10000.0,    # carrying capacity (beetles / m²)
    "T_grow":      5.0,    # avg-temp pivot: growth = 0 below, = r above  (°C)
    "grow_steep":  3.0,    # sigmoid steepness  (°C)

    # Swarming / flight boost — Gaussian peak at optimal flight temperature
    "beta":        0.20,   # swarming amplification strength
    "temp_opt":   20.0,    # optimal temperature for beetle flight  (°C)
    "temp_width":  8.0,    # Gaussian half-width  (°C)

    # Harvest predation
    # current-month harvest (from posek_processed target column)
    "alpha":       0.30,   # predation fraction per normalised current harvest
    "lag":         1,      # harvest effect lag  (months)

    # next-month expected harvest (h1 from target.csv)
    "alpha_h":     0.10,   # predation fraction per normalised forward harvest
                           # smaller than alpha — forward signal adds pressure
                           # but beetles can only be killed by actual cutting

    # Winter mortality — smooth, uses the same sigmoid as growth (inverted).
    "mu_max":      0.95,   # maximum monthly mortality fraction (cold limit)

    # Spatial diffusion
    "gamma":       0.05,   # neighbour diffusion rate

    # Proportional noise (realistic: small when population is low)
    "noise_rel":   0.05,   # noise std as fraction of current B
    "noise_floor": 3.0,    # minimum noise std  (beetles / m²)
}

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recover_harvest(series: pd.Series) -> pd.Series:
    """
    Convert target/horizon column to raw kubikov units.
    If LOG_TARGET=True the values are log1p-transformed; apply expm1 to recover.
    Clips negatives to 0 before and after to handle NaN-filled rows.
    """
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    if LOG_TARGET:
        vals = np.expm1(vals)
    return vals.clip(lower=0.0)


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

    # ── 1. Odsek list and current-month harvest from posek_processed ──────────
    print("Loading posek_processed data …")
    posek = pd.read_csv(POSEK_IN, low_memory=False)
    posek["odsek_id"] = posek["odsek"].astype(str).str.strip()
    posek["leto_mesec"] = posek["leto_mesec"].astype(str).str.strip()

    # Recover kubikov from log1p space if needed
    posek["harvest"] = _recover_harvest(posek["target"])

    odseks = sorted(posek["odsek_id"].unique())
    N = len(odseks)
    odsek_idx = {od: i for i, od in enumerate(odseks)}
    print(f"  Unique odseks: {N:,}  |  LOG_TARGET={LOG_TARGET}")

    # Time axis: infer from posek data
    all_lm   = sorted(posek["leto_mesec"].unique())
    dates    = pd.period_range(all_lm[0], all_lm[-1], freq="M")
    T_steps  = len(dates)
    date_strs = [str(p) for p in dates]
    date_idx  = {s: t for t, s in enumerate(date_strs)}
    print(f"  Time steps: {T_steps}  ({date_strs[0]} → {date_strs[-1]})")

    # ── 2. Current-month harvest matrix  (N × T_steps), normalised per odsek ──
    print("Building current harvest matrix …")
    posek["pi"] = posek["odsek_id"].map(odsek_idx)
    posek["ti"] = posek["leto_mesec"].map(date_idx)
    valid = posek.dropna(subset=["pi", "ti"])

    harvest = np.zeros((N, T_steps), dtype=np.float64)
    harvest[
        valid["pi"].values.astype(int),
        valid["ti"].values.astype(int),
    ] = valid["harvest"].values

    row_max = harvest.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    harvest_norm = harvest / row_max
    print(f"  Current harvest entries mapped: {len(valid):,}")

    # ── 3. Forward harvest matrix from target.csv (h1 = next-month harvest) ───
    print("Loading target.csv for forward harvest signal (h1) …")
    targets = pd.read_csv(TARGET_IN, low_memory=False)
    targets["odsek_id"]  = targets["odsek"].astype(str).str.strip()
    targets["leto_mesec"] = targets["leto_mesec"].astype(str).str.strip()

    # h1 is next-month expected harvest; recover from log1p if needed
    targets["h1_harvest"] = _recover_harvest(targets["h1"])

    targets["pi"] = targets["odsek_id"].map(odsek_idx)
    targets["ti"] = targets["leto_mesec"].map(date_idx)
    valid_t = targets.dropna(subset=["pi", "ti", "h1_harvest"])

    harvest_h1 = np.zeros((N, T_steps), dtype=np.float64)
    harvest_h1[
        valid_t["pi"].values.astype(int),
        valid_t["ti"].values.astype(int),
    ] = valid_t["h1_harvest"].values

    row_max_h1 = harvest_h1.max(axis=1, keepdims=True)
    row_max_h1[row_max_h1 == 0] = 1.0
    harvest_h1_norm = harvest_h1 / row_max_h1
    print(f"  Forward harvest (h1) entries mapped: {len(valid_t):,}")

    # ── 4. Station mapping per odsek ─────────────────────────────────────────
    print("Loading station mapping …")
    postaje = pd.read_csv(POSTAJE_IN, low_memory=False)
    postaje["odsek_id"] = postaje["odsek_id"].astype(str).str.strip()
    postaje = postaje.drop_duplicates(subset=["odsek_id"]).set_index("odsek_id")

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

    # ── 5. Load and aggregate weather ────────────────────────────────────────
    print("Loading weather data …")
    station_weather, g_avg, g_max, g_min = load_monthly_weather(VREME_IN, date_idx)

    # ── 6. Build temperature matrices  (N × T_steps) ─────────────────────────
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

    # ── 7. Spatial neighbour graph ────────────────────────────────────────────
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

    # ── 8. Simulation ─────────────────────────────────────────────────────────
    p           = PARAMS
    K           = p["K"]
    r           = p["r"]
    T_grow      = p["T_grow"]
    grow_steep  = p["grow_steep"]
    beta        = p["beta"]
    t_opt       = p["temp_opt"]
    t_wid       = p["temp_width"]
    alpha       = p["alpha"]
    lag         = p["lag"]
    alpha_h     = p["alpha_h"]
    mu_max      = p["mu_max"]
    gamma       = p["gamma"]
    noise_rel   = p["noise_rel"]
    noise_floor = p["noise_floor"]

    # Seed in winter range: 20–100 beetles/m²
    B = rng.uniform(20.0, 100.0, size=N)

    B_history = np.empty((N, T_steps), dtype=np.float32)

    print(f"\nRunning simulation: {T_steps} months × {N:,} odseks …")
    for t in range(T_steps):

        # ── temperature gate (shared by growth and mortality) ────────────────
        grow_gate = 1.0 / (1.0 + np.exp(-(avg_T[:, t] - T_grow) / grow_steep))

        # ── logistic growth (zero in cold months) ────────────────────────────
        growth = r * grow_gate * B * (1.0 - B / K)

        # ── swarming / flight boost (Gaussian peak at temp_opt) ──────────────
        swarm_factor = np.exp(-((avg_T[:, t] - t_opt) ** 2) / (2.0 * t_wid ** 2))
        swarm_boost  = beta * swarm_factor * B

        # ── winter mortality (inverse of growth gate) ─────────────────────────
        winter_loss = mu_max * (1.0 - grow_gate) * B

        # ── current harvest predation (lagged) ───────────────────────────────
        lag_t     = max(0, t - lag)
        predation = alpha * harvest_norm[:, lag_t] * B

        # ── forward harvest predation: next-month expected cutting (h1) ──────
        # Represents reduced bark-beetle activity when felling is imminent.
        forward_predation = alpha_h * harvest_h1_norm[:, t] * B

        # ── spatial neighbour diffusion ───────────────────────────────────────
        if W_norm.nnz > 0:
            mean_nbr_B = np.asarray(W_norm.dot(B)).ravel()
        else:
            mean_nbr_B = B
        diffusion = gamma * (mean_nbr_B - B)

        # ── proportional noise ────────────────────────────────────────────────
        noise_scale = np.maximum(noise_rel * B, noise_floor)
        noise       = rng.normal(0.0, 1.0, size=N) * noise_scale

        # ── update & clamp ────────────────────────────────────────────────────
        B = (B + growth + swarm_boost
               - winter_loss - predation - forward_predation
               + diffusion + noise)
        B = np.clip(B, 0.0, 3.0 * K)

        B_history[:, t] = B.astype(np.float32)

        if t == 0 or (t + 1) % 36 == 0:
            print(
                f"  {date_strs[t]}:  "
                f"mean={B.mean():.1f}  "
                f"median={np.median(B):.1f}  "
                f"max={B.max():.0f}"
            )

    # ── 9. Assemble & save output ─────────────────────────────────────────────
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
