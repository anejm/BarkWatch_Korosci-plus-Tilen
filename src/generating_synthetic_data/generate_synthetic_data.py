"""
synthetic_pipeline.py
---------------------
Generates synthetic monthly bark beetle population data per (ggo, odsek)
for 2007-01 to 2025-12 using a temperature-driven logistic growth model
with Gaussian temperature suitability, seasonal and outbreak cycles,
multi-lag wood cutting predation, spatial diffusion, and winter mortality.

Model (per parcel, per month):
  1. Temperature suitability: Gaussian response on (avg+min+max)/3
  2. Growth rate:  r_t = r_base * temp_suitability * seasonal * outbreak
  3. Logistic growth:  B += r_t * B * (1 - B/K)
  4. Predator effect:  B -= alpha * wood_cut_lagged * B
  5. Spatial spread:   B += gamma * neighbor_weight * (neighbor_B - B)
  6. Winter mortality: B *= (1 - winter_mortality_rate) if avg_T < threshold
  7. Noise:            B += N(0, noise_std)

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

RNG_SEED = 42


# ---------------------------------------------------------------------------
# Temperature suitability
# ---------------------------------------------------------------------------

def temperature_suitability(
    avg_temp,
    min_temp,
    max_temp,
    optimal_temp=22,
    temp_width=10
):
    """
    Gaussian-like temperature response.
    """

    temp_mean = (
        avg_temp +
        min_temp +
        max_temp
    ) / 3

    suitability = np.exp(
        -((temp_mean - optimal_temp) ** 2)
        / (2 * temp_width ** 2)
    )

    return suitability


# ---------------------------------------------------------------------------
# Seasonal reproduction cycle
# ---------------------------------------------------------------------------

def seasonal_factor(t, amplitude=0.3):
    """
    Annual sinusoidal cycle.
    """

    month = t % 12

    return 1 + amplitude * np.sin(
        2 * np.pi * month / 12
    )


# ---------------------------------------------------------------------------
# Multi-year outbreak cycle
# ---------------------------------------------------------------------------

def outbreak_cycle(
    t,
    outbreak_period_months=96,
    amplitude=0.5
):
    """
    Long-term outbreak waves
    (~5–10 year ecological cycles).
    """

    return 1 + amplitude * np.sin(
        2 * np.pi * t / outbreak_period_months
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_bark_beetles_extended(
    df,
    initial_population=1000,

    # growth parameters
    r_base=0.25,
    carrying_capacity=50000,

    # predator parameters
    alpha=0.00002,

    # spatial interaction
    gamma=0.05,
    neighbor_weight=0.3,

    # lagged predator effect
    wood_lag_weights=(0.6, 0.4),

    # mortality
    winter_temp_threshold=0,
    winter_mortality_rate=0.2,

    # outbreak cycle
    outbreak_period_months=96,

    # noise
    noise_std=200,

    seed=None
):
    """
    Generates synthetic bark beetle population.

    Requires DataFrame columns:

        avg_temperature_month
        min_temperature_month
        max_temperature_month
        number_of_wood_cut

    Optional:

        neighbour_number_of_bark_beetles
        neighbour_number_of_wood_cut

    Returns:
        pandas.Series
    """

    if seed is not None:
        np.random.seed(seed)

    n = len(df)

    bark_beetles = np.zeros(n)
    bark_beetles[0] = initial_population

    for t in range(1, n):

        B_prev = bark_beetles[t - 1]

        # -----------------------------------------
        # Temperature effect
        # -----------------------------------------

        temp_factor = temperature_suitability(
            df.loc[t, "avg_temperature_month"],
            df.loc[t, "min_temperature_month"],
            df.loc[t, "max_temperature_month"]
        )

        r_t = (
            r_base
            * temp_factor
            * seasonal_factor(t)
            * outbreak_cycle(
                t,
                outbreak_period_months
            )
        )

        # -----------------------------------------
        # Logistic growth
        # -----------------------------------------

        growth = (
            r_t
            * B_prev
            * (1 - B_prev / carrying_capacity)
        )

        # -----------------------------------------
        # Multi-lag wood cutting effect
        # -----------------------------------------

        lag1 = (
            df.loc[t - 1,
                   "number_of_wood_cut"]
            if t - 1 >= 0
            else 0
        )

        lag2 = (
            df.loc[t - 2,
                   "number_of_wood_cut"]
            if t - 2 >= 0
            else 0
        )

        wood_cut = (
            wood_lag_weights[0] * lag1 +
            wood_lag_weights[1] * lag2
        )

        predator_effect = (
            alpha
            * wood_cut
            * B_prev
        )

        # -----------------------------------------
        # Neighbor diffusion
        # -----------------------------------------

        if (
            "neighbour_number_of_bark_beetles"
            in df.columns
        ):

            neighbor_B = df.loc[
                t - 1,
                "neighbour_number_of_bark_beetles"
            ]

        elif (
            "neighbour_number_of_wood_cut"
            in df.columns
        ):

            neighbor_B = (
                df.loc[
                    t - 1,
                    "neighbour_number_of_wood_cut"
                ]
                * 10
            )

        else:

            neighbor_B = 0

        spatial_effect = (
            gamma
            * neighbor_weight
            * (neighbor_B - B_prev)
        )

        # -----------------------------------------
        # Winter mortality
        # -----------------------------------------

        avg_temp = df.loc[
            t,
            "avg_temperature_month"
        ]

        winter_factor = 1

        if avg_temp < winter_temp_threshold:

            winter_factor = (
                1 - winter_mortality_rate
            )

        # -----------------------------------------
        # Noise
        # -----------------------------------------

        noise = np.random.normal(
            0,
            noise_std
        )

        # -----------------------------------------
        # Final update
        # -----------------------------------------

        B_new = (
            B_prev
            + growth
            - predator_effect
            + spatial_effect
        )

        B_new *= winter_factor

        B_new += noise

        # prevent negatives
        B_new = max(0, B_new)

        bark_beetles[t] = B_new

    return pd.Series(
        bark_beetles,
        name="number_of_bark_beetles"
    )


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

    parcel_idx: dict[tuple[str, str], int] = {
        (row.ggo, row.odsek): i for i, row in parcels.iterrows()
    }

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
    row_sums[row_sums == 0] = 1.0
    W_norm = diags(1.0 / row_sums).dot(W)
    print(f"  Edges: {W.nnz:,}")

    # ── 4. Time range ────────────────────────────────────────────────────────
    dates     = pd.period_range("2007-01", "2025-12", freq="M")
    T_steps   = len(dates)
    date_strs = [str(p) for p in dates]
    date_idx  = {s: t for t, s in enumerate(date_strs)}

    # ── 5. Weather data ──────────────────────────────────────────────────────
    print("Loading weather data...")
    vreme_all = pd.read_csv(VREME_IN, low_memory=False)
    has_min_temp = "povp_T_min" in vreme_all.columns
    has_max_temp = "povp_T_max" in vreme_all.columns

    vreme = vreme_all[
        ["station_id", "leto_mesec", "povp_T_avg"]
        + (["povp_T_min"] if has_min_temp else [])
        + (["povp_T_max"] if has_max_temp else [])
    ].copy()
    vreme["station_id"] = vreme["station_id"].astype(int)
    global_mean_T = float(np.nanmean(vreme["povp_T_avg"].values))
    print(f"  Global mean temperature: {global_mean_T:.2f} °C")

    # Station assigned to each parcel
    parcel_station = np.full(N, -1, dtype=np.int64)
    postaje_idx = postaje.set_index(["ggo", "odsek_id"])
    for (ggo, odsek), i in parcel_idx.items():
        key = (ggo, odsek)
        if key in postaje_idx.index:
            row = postaje_idx.loc[key]
            sid = row["station_23"] if pd.notna(row["station_23"]) else row["station_123"]
            if pd.notna(sid):
                parcel_station[i] = int(sid)

    unique_stations = set(parcel_station[parcel_station >= 0].tolist())

    station_avg: dict[int, np.ndarray] = {
        sid: np.full(T_steps, global_mean_T, dtype=np.float32)
        for sid in unique_stations
    }
    station_min: dict[int, np.ndarray] = {
        sid: np.full(T_steps, global_mean_T, dtype=np.float32)
        for sid in unique_stations
    }
    station_max: dict[int, np.ndarray] = {
        sid: np.full(T_steps, global_mean_T, dtype=np.float32)
        for sid in unique_stations
    }

    for _, row in vreme.iterrows():
        sid = int(row["station_id"])
        lm  = row["leto_mesec"]
        if sid not in station_avg or lm not in date_idx:
            continue
        t = date_idx[lm]
        if pd.notna(row["povp_T_avg"]):
            avg_val = float(row["povp_T_avg"])
            station_avg[sid][t] = avg_val
            station_min[sid][t] = float(row["povp_T_min"]) if has_min_temp and pd.notna(row.get("povp_T_min")) else avg_val
            station_max[sid][t] = float(row["povp_T_max"]) if has_max_temp and pd.notna(row.get("povp_T_max")) else avg_val

    print("Building temperature matrices...")
    avg_matrix = np.full((N, T_steps), global_mean_T, dtype=np.float32)
    min_matrix = np.full((N, T_steps), global_mean_T, dtype=np.float32)
    max_matrix = np.full((N, T_steps), global_mean_T, dtype=np.float32)

    for i in range(N):
        sid = parcel_station[i]
        if sid >= 0 and sid in station_avg:
            avg_matrix[i] = station_avg[sid]
            min_matrix[i] = station_min[sid]
            max_matrix[i] = station_max[sid]

    # ── 6. Harvest data ──────────────────────────────────────────────────────
    print("Loading harvest data...")
    posek = pd.read_csv(
        POSEK_IN,
        usecols=["ggo", "odsek", "leto_mesec", "rolling_mean_3"],
        low_memory=False,
    )
    posek["ggo"]           = posek["ggo"].astype(str)
    posek["odsek"]         = posek["odsek"].astype(str)
    posek["rolling_mean_3"] = pd.to_numeric(posek["rolling_mean_3"], errors="coerce").fillna(0.0)

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
    ] = valid_posek["rolling_mean_3"].values.astype(np.float32)
    print(f"  Harvest entries mapped: {len(valid_posek):,}")

    # ── 7. Neighbour wood cut (weighted average over spatial neighbours) ──────
    print("Computing neighbour wood cut...")
    neighbour_harvest = np.zeros((N, T_steps), dtype=np.float32)
    if W_norm.nnz > 0:
        for t in range(T_steps):
            neighbour_harvest[:, t] = W_norm.dot(harvest_matrix[:, t])

    # ── 8. Per-parcel simulation ─────────────────────────────────────────────
    print(f"\nRunning simulation: {T_steps} months × {N:,} parcels...")
    B_history = np.zeros((N, T_steps), dtype=np.float32)

    for i in range(N):
        if i % 1000 == 0 and i > 0:
            print(f"  Parcel {i:,}/{N:,}")

        parcel_df = pd.DataFrame({
            "avg_temperature_month":      avg_matrix[i],
            "min_temperature_month":      min_matrix[i],
            "max_temperature_month":      max_matrix[i],
            "number_of_wood_cut":         harvest_matrix[i],
            "neighbour_number_of_wood_cut": neighbour_harvest[i],
        })

        initial_pop = float(rng.uniform(0.02, 0.15) * 50000)

        result_series = generate_bark_beetles_extended(
            parcel_df,
            initial_population=initial_pop,
            seed=RNG_SEED + i,
        )

        B_history[i] = result_series.values.astype(np.float32)

    print("  Done.")

    # ── 9. Assemble output DataFrame ─────────────────────────────────────────
    print("\nAssembling output...")

    ggo_out   = np.repeat(parcels["ggo"].values,   T_steps)
    odsek_out = np.repeat(parcels["odsek"].values, T_steps)
    lm_out    = np.tile(date_strs, N)
    count_out = np.maximum(0, np.round(B_history)).astype(np.int32).ravel()

    result = pd.DataFrame({
        "ggo":               ggo_out,
        "odsek":             odsek_out,
        "leto_mesec":        lm_out,
        "bark_beetle_count": count_out,
    })

    result = result.sort_values(["ggo", "odsek", "leto_mesec"]).reset_index(drop=True)

    # ── 10. Save ─────────────────────────────────────────────────────────────
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
