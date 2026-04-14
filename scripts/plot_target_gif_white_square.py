"""
Generate an animated GIF showing the target feature over time (2007-01 to 2026-12)
for a selected ggo and odsek_id, combining past data and future predictions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PAST_CSV   = Path(__file__).parents[1] / "data/predictions/heatmap_past_data_synthetic.csv"
FUTURE_CSV = Path(__file__).parents[1] / "data/predictions/heatmap_future_predictions_synthetic.csv"
OUTPUT_GIF = Path(__file__).parents[1] / "output/target_over_time.gif"

SELECTED_GGO    = 1        # change to desired ggo
SELECTED_ODSEK  = "01001A" # change to desired odsek_id

FPS         = 12   # frames per second in the GIF
FRAME_PAUSE = 60   # extra pause frames at the end
# ─────────────────────────────────────────────────────────────────────────────


def load_data(ggo, odsek):
    print("Loading past data…")
    past = pd.read_csv(PAST_CSV, low_memory=False)
    past = past[(past["ggo"].astype(str) == str(ggo)) &
                (past["odsek_id"].astype(str) == str(odsek))].copy()

    print("Loading future predictions…")
    future = pd.read_csv(FUTURE_CSV, low_memory=False)
    future = future[(future["ggo"].astype(str) == str(ggo)) &
                    (future["odsek_id"].astype(str) == str(odsek))].copy()

    df = pd.concat([past, future], ignore_index=True)
    df["leto_mesec"] = pd.to_datetime(df["leto_mesec"], format="%Y-%m")
    df = df.sort_values("leto_mesec").reset_index(drop=True)

    # full date range 2007-01 → 2026-12
    full_range = pd.date_range("2007-01", "2026-12", freq="MS")
    df = df.set_index("leto_mesec").reindex(full_range).reset_index()
    df.rename(columns={"index": "leto_mesec"}, inplace=True)

    return df


def build_frames(df):
    """Return list of (dates, values, is_prediction) up to each time step."""
    dates  = df["leto_mesec"].values
    values = df["target"].values
    is_pred = df["is_a_prediction"].fillna(False).astype(bool).values
    frames = []
    for i in range(1, len(dates) + 1):
        frames.append((dates[:i], values[:i], is_pred[:i]))
    return frames


def make_gif(df, ggo, odsek):
    OUTPUT_GIF.parent.mkdir(parents=True, exist_ok=True)

    frames = build_frames(df)
    # Duplicate last frame for pause effect
    frames = frames + [frames[-1]] * FRAME_PAUSE

    all_values = df["target"].dropna().values
    y_min = max(0, all_values.min() * 0.9)
    y_max = all_values.max() * 1.1
    x_min = pd.Timestamp("2007-01-01")
    x_max = pd.Timestamp("2026-12-01")

    # Split index where predictions begin
    pred_start_mask = df["is_a_prediction"].fillna(False).astype(bool)
    pred_start_date = df.loc[pred_start_mask, "leto_mesec"].min() if pred_start_mask.any() else None

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Bark Beetle Target — GGO {ggo} / Odsek {odsek}",
                 color="black", fontsize=13, pad=10)
    ax.set_xlabel("Date", color="black")
    ax.set_ylabel("Target (m³)", color="black")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    # Vertical separator line for prediction start
    if pred_start_date is not None:
        ax.axvline(pred_start_date, color="#ff9900", linewidth=1, linestyle="--", alpha=0.6)
        ax.text(pred_start_date, y_max * 0.97, " forecast →",
                color="#ff9900", fontsize=8, va="top", alpha=0.8)

    line_hist, = ax.plot([], [], color="#4fc3f7", linewidth=1.5, label="Historical")
    line_pred, = ax.plot([], [], color="#ff7043", linewidth=1.5,
                         linestyle="--", label="Predicted")
    dot, = ax.plot([], [], "o", color="white", markersize=4, zorder=5)
    date_text = ax.text(0.01, 0.96, "", transform=ax.transAxes,
                        color="black", fontsize=10, va="top")

    ax.legend(loc="upper left", facecolor="white", labelcolor="black",
              edgecolor="black", fontsize=9)

    def init():
        line_hist.set_data([], [])
        line_pred.set_data([], [])
        dot.set_data([], [])
        date_text.set_text("")
        return line_hist, line_pred, dot, date_text

    def update(frame_idx):
        dates_f, vals_f, preds_f = frames[frame_idx]

        hist_mask = ~preds_f
        pred_mask =  preds_f

        hist_dates = dates_f[hist_mask]
        hist_vals  = vals_f[hist_mask]
        pred_dates = dates_f[pred_mask]
        pred_vals  = vals_f[pred_mask]

        line_hist.set_data(hist_dates, hist_vals)
        line_pred.set_data(pred_dates, pred_vals)

        # Current tip dot
        last_val = vals_f[-1]
        last_date = dates_f[-1]
        if not np.isnan(last_val):
            dot.set_data([last_date], [last_val])
        else:
            dot.set_data([], [])

        ts = pd.Timestamp(last_date)
        date_text.set_text(ts.strftime("%Y-%m"))
        return line_hist, line_pred, dot, date_text

    print(f"Rendering {len(frames)} frames…")
    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=True, interval=1000 // FPS
    )

    print(f"Saving GIF → {OUTPUT_GIF}")
    writer = animation.PillowWriter(fps=FPS)
    ani.save(OUTPUT_GIF, writer=writer)
    plt.close(fig)
    print("Done.")


def main():
    df = load_data(SELECTED_GGO, SELECTED_ODSEK)

    if df["target"].isna().all():
        print(f"No data found for ggo={SELECTED_GGO}, odsek={SELECTED_ODSEK}.")
        print("Available ggo/odsek combinations (first 10):")
        past = pd.read_csv(PAST_CSV, low_memory=False)
        print(past[["ggo", "odsek_id"]].drop_duplicates().head(10).to_string(index=False))
        return

    print(f"Rows loaded: {len(df)}  |  "
          f"Date range: {df['leto_mesec'].min().date()} → {df['leto_mesec'].max().date()}")
    make_gif(df, SELECTED_GGO, SELECTED_ODSEK)


if __name__ == "__main__":
    main()
