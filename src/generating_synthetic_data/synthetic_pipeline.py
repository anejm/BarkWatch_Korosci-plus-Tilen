"""
synthetic_pipeline.py
---------------------
Generates synthetic monthly bark beetle population data per (ggo, odsek)
for 2007-01 to 2025-12 using a temperature-driven logistic growth model
with harvest reduction, spatial diffusion, and seasonality.

Model:
  1. Logistic growth:  B += r(T) * B * (1 - B/K)
  2. Harvest reduction: B *= (1 - alpha * H')
  3. Spatial spread:   B += beta * (W_norm @ B - B)
  4. Noise:            B += N(0, sigma*K)

Inputs:
  data/processed/odseki_processed.csv         – parcel list + area
  data/processed/najblizji_odseki_postaje.csv – spatial neighbours + nearest stations
  data/processed/vreme_mesecno.csv            – monthly temperature per station
  data/processed/posek_processed.csv          – harvest intensity proxy

Output:
  data/synthetic/bark_beetle_population.csv
    Columns: ggo, odsek, leto_mesec, bark_beetle_count
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

ODSEKI_IN  = DATA / "processed" / "odseki_processed.csv"
POSTAJE_IN = DATA / "processed" / "najblizji_odseki_postaje.csv"
VREME_IN   = DATA / "processed" / "vreme_mesecno.csv"
POSEK_IN   = DATA / "processed" / "posek_processed.csv"
OUT_PATH   = DATA / "synthetic" / "bark_beetle_population.csv"

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------
R_MAX    = 0.40   # max monthly intrinsic growth rate
T_OPT    = 22.0   # optimal temperature for growth (°C)
A_SIG    = 0.30   # sigmoid steepness
C_SIG    = 0.50   # sigmoid offset so cold months give negative r
GAMMA    = 0.30   # seasonality amplitude (peaks June–July)
ALPHA    = 0.70   # harvest impact strength (0–1)
BETA     = 0.05   # spatial spread rate
SIGMA    = 0.04   # noise std as fraction of K
K_BASE   = 500.0  # carrying capacity per hectare (arbitrary units)
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Growth rate
# ---------------------------------------------------------------------------

def growth_rate(T: np.ndarray, month: int) -> np.ndarray:
    """
    Temperature- and season-adjusted growth rate.

    r(T) = R_MAX * (sigmoid(T) - C_SIG) * (1 + GAMMA * sin(2pi*(month-4)/12))

    Peaks in June (month=6) via the sin shift of -4 from index origin 1.
    Returns negative values in cold months (T << T_OPT).
    """
    sigmoid = 1.0 / (1.0 + np.exp(-A_SIG * (T - T_OPT)))
    r = R_MAX * (sigmoid - C_SIG)
    season = 1.0 + GAMMA * np.sin(2.0 * np.pi * (month - 4) / 12.0)
    return r * season


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)

    # ── 1. Parcel list ──────────────────────────────────────────────────────
    print("Loading parcel data...")
    parcels = pd.read_csv(ODSEKI_IN, usecols=["ggo", "odsek", "povrsina"])
    parcels["ggo"]   = parcels["ggo"].astype(str)
    parcels["odsek"] = parcels["odsek"].astype(str)
    parcels = parcels.reset_index(drop=True)
    N = len(parcels)
    print(f"  Parcels: {N:,}")

    # Parcel index lookup
    parcel_idx: dict[tuple[str, str], int] = {
        (row.ggo, row.odsek): i for i, row in parcels.iterrows()
    }

    # Carrying capacity proportional to parcel area
    area = parcels["povrsina"].clip(lower=0.1).values.astype(np.float64)
    K = K_BASE * area  # shape (N,)

    # ── 2. Spatial neighbour data ───────────────────────────────────────────
    print("Loading neighbour/station data...")
    postaje = pd.read_csv(
        POSTAJE_IN,
        usecols=["ggo", "odsek_id", "station_23", "station_123", "bliznji_odseki"],
        low_memory=False,
    )
    postaje["ggo"]      = postaje["ggo"].astype(str)
    postaje["odsek_id"] = postaje["odsek_id"].astype(str)

    # ── 3. Build sparse row-normalised weight matrix W_norm ─────────────────
    print("Building spatial neighbour graph...")
    W = lil_matrix((N, N), dtype=np.float32)

    for _, row in postaje.iterrows():
        src_key = (row["ggo"], row["odsek_id"])
        if src_key not in parcel_idx:
            continue
        i = parcel_idx[src_key]
        nbrs_str = row["bliznji_odseki"]
        if pd.isna(nbrs_str) or not nbrs_str:
            continue
        ggo = row["ggo"]
        for nid in str(nbrs_str).split(";"):
            nid = nid.strip()
            if not nid:
                continue
            nkey = (ggo, nid)
            if nkey in parcel_idx:
                j = parcel_idx[nkey]
                if i != j:
                    W[i, j] = 1.0

    W = W.tocsr()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # isolated parcels stay unchanged
    W_norm = diags(1.0 / row_sums).dot(W)  # row-normalised: rows sum to 1 (or 0)
    print(f"  Edges: {W.nnz:,}")

    # ── 4. Pre-build temperature matrix (N × T_steps) ───────────────────────
    # Time range: 2007-01 to 2025-12
    dates       = pd.period_range("2007-01", "2025-12", freq="M")
    T_steps     = len(dates)
    date_strs   = [str(p) for p in dates]
    date_idx    = {s: t for t, s in enumerate(date_strs)}

    print("Loading weather data...")
    vreme = pd.read_csv(
        VREME_IN,
        usecols=["station_id", "leto_mesec", "povp_T_avg"],
        low_memory=False,
    )
    vreme["station_id"] = vreme["station_id"].astype(int)
    global_mean_T = float(np.nanmean(vreme["povp_T_avg"].values))
    print(f"  Global mean temperature: {global_mean_T:.2f} °C")

    # Station ID assigned to each parcel (prefer station_23, fallback station_123)
    parcel_station = np.full(N, -1, dtype=np.int64)
    postaje_idx = postaje.set_index(["ggo", "odsek_id"])
    for (ggo, odsek), i in parcel_idx.items():
        key = (ggo, odsek)
        if key in postaje_idx.index:
            row = postaje_idx.loc[key]
            sid = row["station_23"] if pd.notna(row["station_23"]) else row["station_123"]
            if pd.notna(sid):
                parcel_station[i] = int(sid)

    # Station temperature vectors
    unique_stations = set(parcel_station[parcel_station >= 0].tolist())
    station_temp: dict[int, np.ndarray] = {
        sid: np.full(T_steps, global_mean_T, dtype=np.float32)
        for sid in unique_stations
    }
    for _, row in vreme.iterrows():
        sid = int(row["station_id"])
        lm  = row["leto_mesec"]
        if sid in station_temp and lm in date_idx and pd.notna(row["povp_T_avg"]):
            station_temp[sid][date_idx[lm]] = float(row["povp_T_avg"])

    # Assemble (N × T_steps) temperature matrix
    print("Building temperature matrix...")
    temp_matrix = np.full((N, T_steps), global_mean_T, dtype=np.float32)
    for i in range(N):
        sid = parcel_station[i]
        if sid >= 0 and sid in station_temp:
            temp_matrix[i] = station_temp[sid]

    # ── 5. Pre-build harvest matrix (N × T_steps) ───────────────────────────
    print("Loading and normalising harvest data...")
    posek = pd.read_csv(
        POSEK_IN,
        usecols=["ggo", "odsek", "leto_mesec", "rolling_mean_3"],
        low_memory=False,
    )
    posek["ggo"]           = posek["ggo"].astype(str)
    posek["odsek"]         = posek["odsek"].astype(str)
    posek["rolling_mean_3"] = pd.to_numeric(posek["rolling_mean_3"], errors="coerce").fillna(0.0)

    # Normalize to [0, 1] per parcel
    parcel_max = posek.groupby(["ggo", "odsek"])["rolling_mean_3"].transform("max")
    posek["H_prime"] = posek["rolling_mean_3"] / (parcel_max + 1e-9)
    posek["H_prime"] = posek["H_prime"].fillna(0.0).clip(0.0, 1.0)

    # Map (ggo, odsek) → parcel index
    parcels_idx_series = pd.Series(
        range(N),
        index=pd.MultiIndex.from_arrays([parcels["ggo"], parcels["odsek"]]),
    )
    posek["parcel_i"] = parcels_idx_series.reindex(
        pd.MultiIndex.from_arrays([posek["ggo"], posek["odsek"]])
    ).values
    posek["time_t"] = posek["leto_mesec"].map(date_idx)

    valid_posek = posek.dropna(subset=["parcel_i", "time_t"])
    harvest_matrix = np.zeros((N, T_steps), dtype=np.float32)
    harvest_matrix[
        valid_posek["parcel_i"].values.astype(int),
        valid_posek["time_t"].values.astype(int),
    ] = valid_posek["H_prime"].values.astype(np.float32)
    print(f"  Harvest entries mapped: {len(valid_posek):,}")

    # ── 6. Initialise population ─────────────────────────────────────────────
    # Seed at 2–15 % of carrying capacity with spatial heterogeneity
    B = rng.uniform(0.02, 0.15, size=N) * K

    # ── 7. Simulation ────────────────────────────────────────────────────────
    print(f"\nRunning simulation: {T_steps} months × {N:,} parcels...")

    B_history = np.empty((N, T_steps), dtype=np.float32)

    for t, period in enumerate(dates):
        month = period.month

        # Temperature-driven growth with seasonality
        T_arr = temp_matrix[:, t].astype(np.float64)
        r = growth_rate(T_arr, month)

        # 1. Logistic growth
        B += r * B * (1.0 - B / K)

        # 2. Harvest reduction
        H = harvest_matrix[:, t].astype(np.float64)
        B *= 1.0 - ALPHA * H

        # 3. Spatial diffusion: spread toward neighbour average
        if W_norm.nnz > 0:
            B += BETA * (W_norm.dot(B) - B)

        # 4. Gaussian noise (relative to carrying capacity)
        B += rng.normal(0.0, SIGMA * K)

        # 5. Clamp to [0, 3*K] — no negatives, no runaway explosions
        B = np.clip(B, 0.0, 3.0 * K)

        B_history[:, t] = B.astype(np.float32)

        if (t + 1) % 36 == 0 or t == 0:
            print(
                f"  {date_strs[t]}: "
                f"mean={B.mean():.1f}  "
                f"median={np.median(B):.1f}  "
                f"max={B.max():.0f}"
            )

    # ── 8. Assemble output DataFrame ─────────────────────────────────────────
    print("\nAssembling output...")

    # Axis order: (parcel, time) → flatten in parcel-major order
    # so each parcel's time series is contiguous, matching posek_processed layout
    ggo_out   = np.repeat(parcels["ggo"].values,   T_steps)
    odsek_out = np.repeat(parcels["odsek"].values, T_steps)
    lm_out    = np.tile(date_strs, N)
    count_out = np.maximum(0, np.round(B_history)).astype(np.int32).ravel()

    result = pd.DataFrame({
        "ggo":              ggo_out,
        "odsek":            odsek_out,
        "leto_mesec":       lm_out,
        "bark_beetle_count": count_out,
    })

    # Sort to match canonical (ggo, odsek, leto_mesec) ordering
    result = result.sort_values(["ggo", "odsek", "leto_mesec"]).reset_index(drop=True)

    # ── 9. Save ──────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\nSaved: {OUT_PATH}")
    print(f"  Rows:    {len(result):,}")
    print(f"  Columns: {list(result.columns)}")
    print(f"\nDescriptive stats (bark_beetle_count):")
    print(result["bark_beetle_count"].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))
    print(f"\nSample (first 12 rows for a single parcel):")
    first_parcel = result.groupby(["ggo", "odsek"]).first().index[0]
    sample = result[(result["ggo"] == first_parcel[0]) & (result["odsek"] == first_parcel[1])]
    print(sample.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
