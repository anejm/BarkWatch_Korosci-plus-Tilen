"""
predict_the_future.py
---------------------
Loads the current-state snapshot produced by extract_current_day_data.py
(one row per (ggo, odsek), the most recent observed month) and runs the
trained sequential models to produce genuinely future h1..h12 predictions.

The h{i}_pred value corresponds to the harvest i months beyond the
base leto_mesec for that odsek.

Inputs:
  models/xgb_models.pkl
  data/processed/current_state.csv   ← run extract_current_day_data.py first

Output:
  data/predictions/future_predictions.csv
    Columns: ggo, odsek, leto_mesec, h1_pred .. h12_pred
    Values are in m³ (expm1 applied, matching predictions.csv convention).
    leto_mesec is the base month; h{i}_pred is the predicted harvest
    for base_month + i months.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
CURRENT_PATH = ROOT / "data" / "processed" / "current_state.csv"
MODEL_PATH   = ROOT / "models" / "xgb_models.pkl"
PRED_DIR     = ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = PRED_DIR / "future_predictions.csv"

# ---------------------------------------------------------------------------
# Scenario config
# ---------------------------------------------------------------------------

# Rainfall features scaled by a multiplier (malo=×0.25, veliko=×2.0)
PADAVINE_COLS = [
    "leto_padavine_skupaj_mm",
    "leto_padavine_avg_mm",
    "leto_dni_s_padavinami",
    "leto_snezna_odeja_max_cm",
    "leto_novi_sneg_skupaj_cm",
    "leto_dni_s_snegom",
]
PADAVINE_SCALE = {"malo": 0.25, "normalno": 1.0, "veliko": 2.0}

# Temperature features shifted by an offset in °C (nizko=−3 °C, visoko=+3 °C)
TEMPERATURA_COLS = [
    "leto_povp_T_avg",
    "leto_max_T_mesec",
    "leto_min_T_mesec",
]
TEMPERATURA_OFFSET = {"nizko": -3.0, "normalno": 0.0, "visoko": 3.0}


# ---------------------------------------------------------------------------
# Column config  (must match train.py / testing.py)
# ---------------------------------------------------------------------------
POSSIBLE_KEYS = ["ggo", "odsek", "odsek_id", "leto_mesec"]
DROP_COLS     = [
    "datum", "leto", "target", "log1p_target",
    "sosedi_target_sum", "sosedi_target_mean",
    "sosedi_target_std", "sosedi_target_median",
    "sosedi_log1p_target_mean",
]
TARGET_COLS = [f"h{h}" for h in range(1, 13)]
PRED_COLS   = [f"h{h}_pred" for h in range(1, 13)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (id_df, X_base), filling NaN features with 0."""
    id_cols   = [c for c in POSSIBLE_KEYS if c in df.columns]
    drop_cols = [c for c in DROP_COLS + TARGET_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in set(id_cols + drop_cols)]

    id_df = df[id_cols].reset_index(drop=True)
    X     = df[feat_cols].reset_index(drop=True)

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        print(f"  Filling NaN in {len(nan_cols)} columns with 0")
        X = X.fillna(0)

    return id_df, X


def _two_stage_predict(clf, reg, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    clf_prob = clf.predict_proba(X)[:, 1]
    clf_bin  = (clf_prob >= threshold).astype(int)
    reg_pred = np.maximum(reg.predict(X), 0)
    return clf_bin * reg_pred


def predict(models: dict, X_base: pd.DataFrame, h1_multiplier: float = 1.0) -> np.ndarray:
    """
    Run sequential inference: h_i model receives base features + pred_h1..pred_h(i-1).
    Returns array of shape (n_rows, 12).

    h1_multiplier: scale factor applied to the h1 prediction before it is
                   fed into subsequent models (and stored in the output).
    """
    X_aug = X_base.copy()
    preds = np.zeros((len(X_base), 12))

    for i, col in enumerate(TARGET_COLS):
        m            = models[col]
        feature_cols = m["feature_cols"]
        threshold    = m.get("threshold", 0.5)

        # Add any missing prediction columns as zeros
        for fc in feature_cols:
            if fc not in X_aug.columns:
                X_aug[fc] = 0.0

        X_aligned   = X_aug[feature_cols]
        preds[:, i] = _two_stage_predict(m["clf"], m["reg"], X_aligned, threshold)

        # Apply h1 multiplier to the first horizon prediction
        if i == 0 and h1_multiplier != 1.0:
            preds[:, i] = preds[:, i] * h1_multiplier

        if i < 11:
            X_aug = X_aug.copy()
            X_aug[f"pred_{col}"] = preds[:, i]

    return preds


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate future harvest predictions with optional scenario overrides."
    )
    parser.add_argument(
        "--h1",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Multiply the h1 prediction from model 1 by this factor (0.0–5.0) "
            "before feeding it into subsequent models. Default: 1.0 (no change)."
        ),
    )
    parser.add_argument(
        "--padavine",
        choices=["malo", "normalno", "veliko"],
        default="normalno",
        help=(
            "Rainfall scenario: 'malo' scales rainfall features to ×0.25, "
            "'normalno' leaves them unchanged, 'veliko' scales them to ×2.0."
        ),
    )
    parser.add_argument(
        "--temperatura",
        choices=["nizko", "normalno", "visoko"],
        default="normalno",
        help=(
            "Temperature scenario: 'nizko' subtracts 3 °C from temperature features, "
            "'normalno' leaves them unchanged, 'visoko' adds 3 °C."
        ),
    )
    args = parser.parse_args()
    if not (0.0 <= args.h1 <= 5.0):
        parser.error(f"--h1 must be between 0.0 and 5.0, got {args.h1}")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"Loading models from {MODEL_PATH} …")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    models: dict = joblib.load(MODEL_PATH)
    print(f"  Loaded {len(models)} horizon models: {list(models.keys())}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(
            f"Current state file not found: {CURRENT_PATH}\n"
            "Run extract_current_day_data.py first."
        )
    print(f"\nLoading current state from {CURRENT_PATH.name} …")
    current = pd.read_csv(CURRENT_PATH, low_memory=False)
    print(f"  Rows: {len(current):,}")
    print(f"  Base months range: {current['leto_mesec'].min()} → {current['leto_mesec'].max()}")

    id_df, X_base = prepare_features(current)
    print(f"  Base features: {X_base.shape[1]}")

    # ------------------------------------------------------------------
    # Apply scenario overrides
    # ------------------------------------------------------------------
    if args.padavine != "normalno":
        scale = PADAVINE_SCALE[args.padavine]
        cols_present = [c for c in PADAVINE_COLS if c in X_base.columns]
        X_base[cols_present] = X_base[cols_present] * scale
        print(f"\nPadavine scenario '{args.padavine}': scaled {len(cols_present)} rainfall features by ×{scale}")

    if args.temperatura != "normalno":
        offset = TEMPERATURA_OFFSET[args.temperatura]
        cols_present = [c for c in TEMPERATURA_COLS if c in X_base.columns]
        X_base[cols_present] = X_base[cols_present] + offset
        print(f"Temperatura scenario '{args.temperatura}': shifted {len(cols_present)} temperature features by {offset:+.1f} °C")

    if args.h1 != 1.0:
        print(f"H1 multiplier: ×{args.h1}")

    print("\nRunning sequential inference …")
    preds = predict(models, X_base, h1_multiplier=args.h1)

    # Convert from log1p space to m³ (matching predictions.csv convention)
    preds_m3 = np.expm1(np.maximum(preds, 0))

    out = pd.concat(
        [id_df, pd.DataFrame(preds_m3, columns=PRED_COLS)],
        axis=1,
    )
    out.to_csv(OUT_PATH, index=False)
    print(f"\nFuture predictions saved → {OUT_PATH}  ({len(out):,} rows, values in m³)")
    print("  Each h{i}_pred corresponds to base leto_mesec + i months.")


if __name__ == "__main__":
    main()
